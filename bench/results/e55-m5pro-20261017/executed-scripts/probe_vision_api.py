"""Synthetic API/vision checks; caller owns exclusive access and memory guard."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--base', default='http://127.0.0.1:12346')
p.add_argument('--config', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = Path('/Users/jakecantrell/inference-lab/contributions/strata-official-296b59d')
a.out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, '/Users/jakecantrell/inference-lab/contributions/strata-vision-20261017/source/bench/experiments')
from e50_latency_tuning import grade, visible

def append(name, row):
    with (a.out / name).open('a') as f: f.write(json.dumps(row) + '\n')
    if name == 'images.jsonl' and (not row.get('content_correct') or row.get('reuse_pass') is False):
        raise RuntimeError('Image content/cache validation failed: ' + row['case'])

def request(body):
    start = time.perf_counter(); text = ''; ttft = None; final = {}; tool = []
    req = urllib.request.Request(a.base + '/v1/chat/completions', json.dumps(dict(body, stream=True)).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=360) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            item = json.loads(line[6:])
            if item.get('error'): raise RuntimeError(str(item['error']))
            if item.get('timings'): final['timings'] = item['timings']
            if item.get('usage'): final['usage'] = item['usage']
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                content = delta.get('content') or ''
                if content and ttft is None: ttft = time.perf_counter() - start
                text += content
                tool.extend(delta.get('tool_calls') or [])
                if choice.get('finish_reason'): final['finish_reason'] = choice['finish_reason']
    return {'text': visible(text), 'wall_s': time.perf_counter() - start, 'ttft_s': ttft,
            'tool_deltas': tool, **final}

proc = None; log = None
try:
    if a.config:
        log = (a.out / 'server.log').open('w')
        start = time.perf_counter()
        proc = subprocess.Popen([sys.executable, '-m', 'serve.server', '--engine', 'strata', '--config', str(a.config),
                                 '--host', '127.0.0.1', '--port', '12346'], cwd=root, stdout=log, stderr=log)
        deadline = time.monotonic() + 360
        while True:
            if proc.poll() is not None: raise RuntimeError('Official API exited: ' + str(proc.returncode))
            try:
                with urllib.request.urlopen(a.base + '/health', timeout=2) as response:
                    data = json.load(response)
                if data.get('status') == 'ok': break
            except (OSError, urllib.error.URLError): pass
            if time.monotonic() > deadline: raise TimeoutError('Official API did not become healthy')
            time.sleep(2)
        append('startup.jsonl', {'seconds': time.perf_counter() - start, 'health': data})
    cases = json.loads((root / 'image-fixtures.json').read_text())
    for repeat in range(2):
        for case in cases:
            print('image round-' + str(repeat) + ': ' + case['name'], flush=True)
            url = 'data:image/png;base64,' + base64.b64encode(Path(case['path']).read_bytes()).decode()
            body = {'model': 'flash-next-uncensored', 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': case['prompt']}, {'type': 'image_url', 'image_url': {'url': url}}]}],
                'max_tokens': 128, 'temperature': 0, 'seed': 42,
                'chat_template_kwargs': {'enable_thinking': False}}
            row = request(body); row.update(case=case['name'], round=repeat)
            row.update(grade(row['text'], case['expected']))
            row['content_correct'] &= row.get('finish_reason') == 'stop'
            row['strict_correct'] &= row.get('finish_reason') == 'stop'
            append('images.jsonl', row)
    first = cases[0]
    reuse_body = {'model':'flash-next-uncensored','messages':[{'role':'user','content':[
        {'type':'text','text':first['prompt']}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+
        base64.b64encode(Path(first['path']).read_bytes()).decode()}}]}],
        'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    for label in ['image-reuse-prime','image-reuse-live','image-reuse-parked']:
        if label == 'image-reuse-parked':
            request({'model':'flash-next-uncensored','messages':[{'role':'user','content':'Reply exactly READY.'}],
                     'max_tokens':8,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
        row=request(reuse_body)
        row.update(case=label, **grade(row['text'],first['expected']))
        row['reuse_pass'] = label == 'image-reuse-prime' or row.get('timings',{}).get('cache_n',0) > 0
        append('images.jsonl',row)
    # Close a streaming image request after visible output, then verify a fresh image request recovers.
    cancel_body=json.loads(json.dumps(reuse_body))
    cancel_body['messages'][0]['content'][0]['text']='Read this image and repeat its printed text 100 times. Keep repeating it.'
    cancel_body.update(max_tokens=512,stream=True)
    req=urllib.request.Request(a.base+'/v1/chat/completions',json.dumps(cancel_body).encode(),{'Content-Type':'application/json'})
    saw_content=False
    with urllib.request.urlopen(req,timeout=180) as response:
        for raw in response:
            if not raw.startswith(b'data: ') or raw.strip()==b'data: [DONE]': continue
            item=json.loads(raw[6:])
            if any(c.get('delta',{}).get('content') for c in item.get('choices',[])):
                saw_content=True
                break
    append('recovery.jsonl',{'case':'image-http-disconnect','closed_after_visible_content':saw_content})
    row=request(reuse_body)
    row.update(case='image-after-disconnect', **grade(row['text'],first['expected']))
    append('images.jsonl',row)
    pair = cases[:2]
    parts = [{'type':'text','text':'Read both synthetic images. Return only a JSON object with keys first and second, containing their printed text in image order.'}]
    for case in pair:
        parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,' + base64.b64encode(Path(case['path']).read_bytes()).decode()}})
    row = request({'model':'flash-next-uncensored','messages':[{'role':'user','content':parts}],
                   'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
    row.update(case='two-images', **grade(row['text'], {'first':'STRATA 731','second':'MAPLE 482'}))
    append('images.jsonl',row)
    first=cases[0]
    conversation=[{'role':'user','content':[{'type':'text','text':first['prompt']},
                 {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(first['path']).read_bytes()).decode()}}]},
                 {'role':'assistant','content':first['expected']},
                 {'role':'user','content':'What text was in that image? Return only that text.'}]
    row=request({'model':'flash-next-uncensored','messages':conversation,'max_tokens':64,'temperature':0,
                 'chat_template_kwargs':{'enable_thinking':False}})
    row.update(case='image-follow-up', **grade(row['text'],first['expected']))
    append('images.jsonl',row)
    # Intervening requests with a different image test that stale image embeddings are not reused.
    ready = request({'model': 'flash-next-uncensored', 'messages': [{'role': 'user', 'content': 'Reply exactly READY.'}],
                     'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
    ready.update(grade(ready['text'], 'READY'))
    append('recovery.jsonl', dict(case='text-after-images', **ready))
    bad = urllib.request.Request(a.base + '/v1/chat/completions', b'{broken', {'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(bad, timeout=10)
        append('recovery.jsonl', {'case': 'malformed-json', 'rejected': False})
    except urllib.error.HTTPError as error:
        append('recovery.jsonl', {'case': 'malformed-json', 'rejected': 400 <= error.code < 500, 'status': error.code})
    again = request({'model': 'flash-next-uncensored', 'messages': [{'role': 'user', 'content': 'Reply exactly READY.'}],
                     'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
    again.update(grade(again['text'], 'READY'))
    append('recovery.jsonl', dict(case='after-malformed-json', **again))
    append('complete.jsonl', {'completed': True})
finally:
    if proc:
        import psutil
        try: descendants = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess: descendants = []
        # Stop the parent first: its shutdown closes the resident engine and encoder.
        proc.terminate()
        try: proc.wait(timeout=45)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        _, alive = psutil.wait_procs(descendants, timeout=5)
        for child in alive:
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(alive, timeout=10)
        for child in alive:
            try: child.kill()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        if alive: raise RuntimeError('Owned HTTP descendants survived cleanup')
        # A parent can fail during startup before its shutdown handler is installed.
        # Reap only this isolated checkout's two known child executables before returning.
        import psutil
        owned = []
        for child in psutil.process_iter(['pid', 'cmdline']):
            command = child.info['cmdline'] or []
            if command and command[0] in ('/Users/jakecantrell/inference-lab/contributions/strata-vision-20261017/mlx-vision', str(root / 'build-vision/bin/strata-vision')):
                owned.append(child)
                try: child.terminate()
                except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(owned, timeout=15)
        for child in alive: child.kill()
        _, alive = psutil.wait_procs(alive, timeout=10)
        if alive: raise RuntimeError('Experimental child did not exit; cannot load another model')
    if log: log.close()
