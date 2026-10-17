#!/usr/bin/env python3
"""Synthetic HTTP/Strata line-protocol comparisons. Never stops the production service."""
import argparse
import ast
import hashlib
import json
import os
import platform
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def post(base, path, payload, timeout=300):
    req = urllib.request.Request(base.rstrip('/') + path, json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def clean(text):
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    if '<think>' in text and '</think>' not in text:
        return ''
    return text.split('</think>')[-1].strip()


def score(text, expected):
    try:
        return json.loads(clean(text)) == expected if isinstance(expected, (dict, list)) else clean(text) == expected
    except (ValueError, TypeError):
        return False


def prepare(base, contexts):
    """Only template/tokenizer API calls; no inference requests."""
    def render(messages):
        return post(base, '/apply-template', {'messages': messages,
                    'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    def encode(text):
        return post(base, '/tokenize', {'content': text, 'add_special': False,
                                       'parse_special': True})['tokens']
    cases = []
    short = [('ready', 'Reply exactly READY.', 'READY'),
             ('arithmetic', 'Return only JSON with answer: 17 sheep, all but 9 leave, '
              'buy twice the remaining number, then sell 5. How many remain?', {'answer': 22}),
             ('invoice', 'Return only JSON with total and delivery_date. Two items cost $12.50 '
              'each and one costs $8. No delivery date is provided.', {'total': 33, 'delivery_date': None})]
    for name, user, expected in short:
        rendered = render([{'role': 'user', 'content': user}])
        cases.append({'name': name, 'kind': 'short', 'rendered': rendered,
                      'ids': encode(rendered), 'expected': expected, 'max_tokens': 128})
    for target in contexts:
        expected = {'start': 'CEDAR-741', 'middle': 'WILLOW-582', 'end': 'PINE-963'}
        def user(n):
            rows = [f'Record {i:06}: ordinary shipment status pending; units {(i * 7) % 97}.\n'
                    for i in range(n)]
            return ('Read the ledger and return only JSON with start, middle, end codes.\n'
                    f'Context target {target}. Start code: CEDAR-741.\n' + ''.join(rows[:n//2])
                    + 'Middle code: WILLOW-582.\n' + ''.join(rows[n//2:])
                    + 'End code: PINE-963. Return the three codes only.')
        low, high = 0, target
        while low < high:
            mid = (low + high) // 2
            if len(encode(render([{'role': 'user', 'content': user(mid)}]))) < target:
                low = mid + 1
            else:
                high = mid
        initial_user = user(low)
        rendered = render([{'role': 'user', 'content': initial_user}])
        marker = 'HARNESS_ASSISTANT_MARKER_8347'
        followup = render([{'role': 'user', 'content': initial_user},
                           {'role': 'assistant', 'content': marker},
                           {'role': 'user', 'content': 'Repeat the same three codes as JSON.'}])
        if followup.count(marker) != 1:
            raise ValueError('Cannot identify assistant history in deployed template')
        prefix, suffix = followup.split(marker)
        interrupt = render([{'role': 'user', 'content': 'Reply exactly TITLE.'}])
        cases.append({'name': f'context-{target}', 'kind': 'conversation', 'rendered': rendered,
                      'ids': encode(rendered), 'expected': expected, 'max_tokens': 128,
                      'resume_prefix': prefix, 'resume_suffix': suffix,
                      'interrupt_rendered': interrupt, 'interrupt_ids': encode(interrupt)})
    return {'schema': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'synthetic_only': True, 'template_source': base, 'thinking': False, 'cases': cases}


class HTTPBackend:
    def __init__(self, base, timeout):
        self.base, self.timeout = base, timeout

    def run(self, rendered, ids=None):
        start = time.monotonic()
        body = {'prompt': rendered, 'n_predict': 128, 'temperature': 0, 'seed': 42,
                'cache_prompt': True, 'stream': True}
        req = urllib.request.Request(self.base.rstrip('/') + '/completion', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        output, first_token, first_visible, final = '', None, None, {}
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                item = json.loads(line[6:])
                content = item.get('content', '')
                if content and first_token is None:
                    first_token = time.monotonic() - start
                output += content
                if clean(output) and first_visible is None:
                    first_visible = time.monotonic() - start
                if item.get('stop') or 'timings' in item:
                    final = item
        if not final:
            raise RuntimeError('HTTP stream ended without a completion frame')
        timings = final.get('timings', {})
        return {'text': output, 'wall_s': time.monotonic()-start, 'ttft_s': first_token,
                'first_visible_s': first_visible, 'reused_tokens': timings.get('cache_n'),
                'prompt_tokens': timings.get('prompt_n'), 'generated_tokens': timings.get('predicted_n'),
                'decode_tps': timings.get('predicted_per_second'), 'timings': timings}

    def close(self):
        pass


def parse_done(line):
    fields = line.split()
    if len(fields) < 15 or fields[0] != 'DONE':
        raise ValueError('Malformed DONE frame')
    return {'generated_tokens': int(fields[1]), 'prompt_tokens': int(fields[2]),
            'prompt_ms': float(fields[3]), 'decode_ms': float(fields[4]), 'finish': fields[5],
            'draft_accepted': int(fields[6]), 'drafted': int(fields[7]), 'reused_tokens': int(fields[8]),
            'read_tokens': int(fields[14]),
            'decode_tps': int(fields[1])*1000/max(float(fields[4]), .001)}


class NativeBackend:
    def __init__(self, command, cwd, encode, decode, timeout, stderr_path):
        self.timeout, self.encode, self.decode = timeout, encode, decode
        self.queue = queue.Queue()
        self.stderr = open(stderr_path, 'w')
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1)
        def reader():
            for line in self.process.stdout:
                self.queue.put(line.strip())
            self.queue.put(None)
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.startup = self.until_ready()
        except BaseException:
            self.close()
            raise

    def next_line(self, deadline):
        try:
            line = self.queue.get(timeout=max(.001, deadline-time.monotonic()))
        except queue.Empty:
            raise TimeoutError('Strata protocol deadline exceeded') from None
        if line is None:
            raise RuntimeError('Strata exited before completing the protocol')
        if line.startswith(('ERR', 'FATAL')):
            raise RuntimeError(line)
        return line

    def until_ready(self):
        deadline, lines = time.monotonic()+self.timeout, []
        while True:
            line = self.next_line(deadline)
            lines.append(line)
            if line.startswith('READY '):
                return lines

    def run(self, rendered, ids=None, max_tokens=128):
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        ids = ids if ids is not None else self.encode(rendered)
        start, output, first_token, first_visible = time.monotonic(), [], None, None
        self.process.stdin.write(f'GEN {max_tokens} temperature=0 seed=42 ' + ','.join(map(str, ids)) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.next_line(start+self.timeout)
            if line.startswith('T '):
                output.append(int(line.split()[1]))
                if first_token is None:
                    first_token = time.monotonic()-start
                if first_visible is None and clean(self.decode(output)):
                    first_visible = time.monotonic()-start
            if line.startswith('DONE '):
                return {'text': clean(self.decode(output)), 'wall_s': time.monotonic()-start,
                        'ttft_s': first_token, 'first_visible_s': first_visible, **parse_done(line)}

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('QUIT\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


class MemorySampler:
    def __init__(self, path):
        self.path, self.stop, self.pid = path, threading.Event(), None
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with open(self.path, 'w') as output:
            while not self.stop.is_set():
                row = {'elapsed_monotonic': time.monotonic()}
                if platform.system() == 'Darwin':
                    for name, command in [('swap', ['sysctl', 'vm.swapusage']), ('vm', ['vm_stat'])]:
                        p = subprocess.run(command, capture_output=True, text=True, timeout=3)
                        row[name] = p.stdout.strip()
                        row[name+'_exit'] = p.returncode
                    if self.pid:
                        p = subprocess.run(['ps', '-o', 'pid=,rss=', '-p', str(self.pid)],
                                           capture_output=True, text=True, timeout=3)
                        row['candidate_rss_kib'] = p.stdout.strip()
                output.write(json.dumps(row)+'\n')
                output.flush()
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=8)


def run_cases(backend, fixtures, repeats, idle_seconds, on_row=None):
    rows = []
    def request(case, phase, rendered, expected, ids=None):
        result = backend.run(rendered, ids)
        text = result.pop('text')
        row = {'case': case['name'], 'phase': phase, 'repeat': repeat,
               'correct': score(text, expected), 'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'synthetic_output': text,
               **result}
        rows.append(row)
        if on_row:
            on_row(row)
        return text
    for repeat in range(repeats):
        for case in fixtures['cases']:
            answer = request(case, 'initial', case['rendered'], case['expected'], case['ids'])
            if case['kind'] == 'conversation':
                resume = case['resume_prefix'] + answer + case['resume_suffix']
                warm_answer = request(case, 'immediate-followup', resume, case['expected'])
                # Continue from the ACTUAL warm answer, so this is a new turn, not a replay of the same prompt.
                prefix = resume + warm_answer
                # The suffix begins with the assistant end marker and the next user turn.
                next_turn = prefix + case['resume_suffix']
                request(case, 'interrupt', case['interrupt_rendered'], 'TITLE', case['interrupt_ids'])
                request(case, 'after-interrupt', next_turn, case['expected'])
            if idle_seconds:
                time.sleep(idle_seconds)
                request(case, 'after-idle', case['rendered'], case['expected'], case['ids'])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--url', default='http://127.0.0.1:12345')
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 10000, 32768])
    p.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--fixtures', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--backend', choices=['http', 'native'], required=True)
    r.add_argument('--url', default='http://127.0.0.1:12345')
    r.add_argument('--command-json', help='JSON array; launch candidate without a shell')
    r.add_argument('--cwd', type=Path)
    r.add_argument('--tokenizer', type=Path)
    r.add_argument('--label', required=True)
    r.add_argument('--revision', required=True)
    r.add_argument('--repeats', type=int, default=3)
    r.add_argument('--idle-seconds', type=float, default=0)
    r.add_argument('--timeout', type=float, default=600)
    r.add_argument('--maintenance-ready', action='store_true',
                   help='Operator has confirmed the photo queue is finished/paused and drained')
    args = parser.parse_args()
    if args.mode == 'prepare':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prepare(args.url, args.contexts), indent=2)+'\n')
        print('Synthetic fixtures prepared; no generation requests sent.')
        return
    if not args.maintenance_ready:
        parser.error('Confirm the workload is finished/paused and drained before --maintenance-ready')
    if args.repeats < 1 or args.idle_seconds < 0 or args.timeout <= 0:
        parser.error('Invalid repeat, idle, or timeout value')
    fixtures = json.loads(args.fixtures.read_text())
    if fixtures.get('synthetic_only') is not True:
        parser.error('Only prepared synthetic fixtures are accepted')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'label': args.label, 'revision': args.revision,
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fixtures_sha256': hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'backend': args.backend, 'repeats': args.repeats, 'idle_seconds': args.idle_seconds,
                'precision_env': os.environ.get('MLX_ENABLE_TF32'), 'status': 'running'}
    backend = None
    try:
        with MemorySampler(args.out/'memory.jsonl') as sampler:
            if args.backend == 'native':
                if not args.command_json or not args.cwd or not args.tokenizer:
                    raise ValueError('Native runs require command-json, cwd, and tokenizer')
                with socket.socket() as sock:
                    sock.settimeout(2)
                    if sock.connect_ex(('127.0.0.1', 12345)) == 0:
                        raise RuntimeError('Production port 12345 is still listening; refusing a second resident model')
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(args.tokenizer))
                encode = lambda text: tok.encode(text, add_special_tokens=False).ids
                for case in fixtures['cases']:
                    if encode(case['rendered']) != case['ids']:
                        raise ValueError('Local tokenizer differs from prepared deployed tokenizer')
                command = json.loads(args.command_json)
                if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                    raise ValueError('command-json must be a nonempty string array')
                if '--conversation-cache-mib' in command:
                    tree = ast.parse((args.cwd/'strata_mlx/serve.py').read_text())
                    known = [ast.literal_eval(n.value) for f in tree.body if isinstance(f, ast.FunctionDef)
                             and f.name == '_options' for n in ast.walk(f) if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'known' for t in n.targets)]
                    if not known or '--conversation-cache-mib' not in known[0]:
                        raise ValueError('Source parser does not implement the requested conversation cache option')
                manifest['command'] = command
                backend = NativeBackend(command, args.cwd, encode,
                                        lambda ids: tok.decode(ids, skip_special_tokens=False),
                                        args.timeout, args.out/'synthetic-engine-stderr.log')
                manifest['startup'] = backend.startup
                sampler.pid = backend.process.pid
            else:
                backend = HTTPBackend(args.url, args.timeout)
            with open(args.out/'requests.jsonl', 'w') as progress:
                def on_row(row):
                    progress.write(json.dumps(row)+'\n')
                    progress.flush()
                manifest['rows'] = run_cases(backend, fixtures, args.repeats, args.idle_seconds, on_row)
            manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error_type'] = type(error).__name__
        raise
    finally:
        if backend:
            backend.close()
        (args.out/'report.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': manifest['status'], 'checks': len(manifest['rows']),
                      'correct': sum(x['correct'] for x in manifest['rows'])}))


if __name__ == '__main__':
    main()
