"""Exact-branch supported-file image qualification through the unchanged official HTTP server.

Caller must own the machine's workload gate, memory guard and service restoration.
No private images: this script creates its own two 1024x768 OCR fixtures.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil
from PIL import Image, ImageDraw, ImageFont


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')


def drop_shards(model):
    """F_RDADVISE is not a drop: use the same F_NOCACHE transition as existing lab protocols."""
    for path in sorted(model.parent.glob('*.gguf')):
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, 48, 1)  # Darwin F_NOCACHE: discard file's cached data
            fcntl.fcntl(fd, 48, 0)
        finally:
            os.close(fd)


def request(base, messages, limit=128):
    body = {'model': 'vision-review-q2', 'messages': messages, 'max_tokens': limit,
            'temperature': 0, 'seed': 42, 'stream': True,
            'chat_template_kwargs': {'enable_thinking': False}}
    start = time.perf_counter()
    text, ttft, end, final = '', None, False, {}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            if raw.strip() == b'data: [DONE]':
                end = True
                continue
            item = json.loads(raw[6:])
            if item.get('error'):
                raise RuntimeError(str(item['error']))
            for key in ('timings', 'usage'):
                if item.get(key):
                    final[key] = item[key]
            for choice in item.get('choices', []):
                content = choice.get('delta', {}).get('content') or ''
                if content and ttft is None:
                    ttft = time.perf_counter() - start
                text += content
                if choice.get('finish_reason'):
                    final['finish_reason'] = choice['finish_reason']
    return dict(text=text.strip(), ttft_s=ttft, wall_s=time.perf_counter()-start,
                stream_done=end, **final)


def fixture(out, text):
    img = Image.new('RGB', (1024, 768), 'white')
    draw = ImageDraw.Draw(img)
    # The font's actual bytes are pinned alongside the fixture pixels in metadata.
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 96)
    draw.text((100, 300), text, fill='black', font=font)
    path = out / (text.lower().replace(' ', '-') + '.png')
    img.save(path)
    return path, font_path


def image_part(path):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                                             base64.b64encode(path.read_bytes()).decode()}}


def complete(row):
    return row.get('stream_done') and row.get('finish_reason') == 'stop'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    fixtures = a.out/'synthetic-fixtures'
    fixtures.mkdir()
    pair = [fixture(fixtures, text) for text in ('STRATA 731', 'MAPLE 482')]
    config = json.loads(a.config.read_text())
    native = config['args']
    model = Path(native[native.index('--native')+1])
    assert model.name == 'Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
    assert '--expert-memory' not in native or native[native.index('--expert-memory')+1] != 'resident'
    source = Path(config['cwd'])
    hashes = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted((source/'strata_mlx').glob('*.py'))}
    meta = {'protocol': 'e56-pr13-supported-file-v2', 'source_sha256': hashes,
            'native_args': native, 'model_files': [{'name': f.name, 'bytes': f.stat().st_size}
                                                   for f in sorted(model.parent.glob('*.gguf'))],
            'fixture_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f, _ in pair},
            'font_sha256': hashlib.sha256(pair[0][1].read_bytes()).hexdigest(),
            'external_server_revision': '296b59d2f73c8f04a805df29c2fa97655adace78',
            'precision': {'dtype':'float16', 'MLX_ENABLE_TF32':os.getenv('MLX_ENABLE_TF32'),
                          'MTL_DEBUG_LAYER':os.getenv('MTL_DEBUG_LAYER')},
            'cold_warm': 'F_NOCACHE requested on both shards before each measured request; actual page residency not measured; live engine/expert arrays retained',
            'comparison_limits': 'Same weights and engine; image presence changes prompt length and positions. Not isolated ImageRoPE cost.'}
    (a.out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    proc = None
    rows = []
    log = (a.out/'server.log').open('w')
    base = 'http://127.0.0.1:12346'
    try:
        proc = subprocess.Popen([sys.executable, '-m', 'serve.server', '--engine', 'strata',
                                 '--config', str(a.config), '--host', '127.0.0.1', '--port', '12346'],
                                cwd=a.official_root, stdout=log, stderr=log)
        deadline = time.monotonic()+420
        while True:
            if proc.poll() is not None:
                raise RuntimeError('HTTP server exited')
            try:
                with urllib.request.urlopen(base+'/health', timeout=3) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('HTTP startup')
            time.sleep(1)

        def run(name, messages, expected=None, limit=128):
            print(name, flush=True)
            drop_shards(model)
            row = request(base, messages, limit)
            try:
                actual = json.loads(row['text']) if isinstance(expected, list) else row['text']
            except ValueError:
                actual = None
            correct = (type(actual) is list and len(actual)==len(expected) and
                       all(type(x) is int for x in actual) and actual==expected) if isinstance(expected,list) else actual==expected
            row.update(case=name, content_correct=complete(row) and correct)
            rows.append(row)
            append(a.out/'api.jsonl', row)
            if not row['content_correct']:
                raise RuntimeError('Incomplete or incorrect '+name)
            return row

        for i, (path, _) in enumerate(pair):
            run('ocr-'+str(i), [{'role':'user','content':[
                {'type':'text','text':'Read the printed text. Reply only with that text.'},image_part(path)]}],
                ('STRATA 731','MAPLE 482')[i])
        # Text alone is >2,048 tokens; images add their actual encoded grid to the measured usage.
        filler = '\n'.join(f'Record {i:03d}: the sample is an ordinary calibration record.' for i in range(190))
        prompt = ('Remember the two image labels in order. Calibration records follow:\n'+filler+
                  '\nIgnore these records. Read both image labels; reply with the first label, a semicolon and a space, then the second label.')
        messages = [{'role':'user','content':[{'type':'text','text':prompt}]+[image_part(f) for f,_ in pair]}]
        row = run('two-images-long', messages, 'STRATA 731; MAPLE 482', 64)
        assert row.get('usage',{}).get('prompt_tokens',0)>2048, row
        messages += [{'role':'assistant','content':row['text']},
                     {'role':'user','content':'What was the SECOND image label? Reply only with that label.'}]
        row = run('two-images-long-followup', messages, 'MAPLE 482', 64)
        assert row.get('usage',{}).get('prompt_tokens',0)>2048, row
        # Shuffled paired decode comparison plus an identical text control on the same live engine.
        instruction = ('Write only the integers 0 through 31 inclusive as a JSON array. '
                       'Do not describe the image or label. Do not use Markdown.')
        expected = list(range(32))
        plan = []
        rng = random.Random(56013)
        for repeat in range(2):
            arms = ['image', 'text', 'control']
            rng.shuffle(arms)
            plan.extend(arms)
        (a.out/'plan.json').write_text(json.dumps(plan)+'\n')
        for i, arm in enumerate(plan):
            # Exact same conversation within each triplet, differing only by image presence.
            # Parking is off for this protocol. An unrelated request replaces live state
            # before each measurement, so the duplicate text control is freshly read.
            run(f'interrupt-{i}', [{'role':'user','content':'Reply exactly READY.'}], 'READY', 8)
            parts=[{'type':'text','text':'The calibration label is STRATA 731. '+instruction}]
            if arm=='image':
                parts.append(image_part(pair[0][0]))
            row=run(f'decode-{i}-{arm}',[
                {'role':'system','content':f'Calibration trial {i//3:03d}. Follow the user instruction.'},
                {'role':'user','content':parts}], expected, limit=160)
            assert row['timings'].get('cache_n', 0) == 0, row
            parsed=json.loads(row['text'])
            assert type(parsed) is list and len(parsed)==32 and all(type(x) is int for x in parsed) and parsed==expected
        run('text-recovery',[{'role':'user','content':'Reply exactly READY.'}], 'READY', 8)
        summary={'complete':True,'requests':len(rows),'correct':sum(r['content_correct'] for r in rows),
                 'long_prompt_tokens':[r['usage']['prompt_tokens'] for r in rows if 'long' in r['case']],
                 'decode_rows':[r for r in rows if r['case'].startswith('decode-')]}
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        if proc:
            try:
                descendants=psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants=[]
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            _, alive=psutil.wait_procs(descendants,timeout=5)
            for child in alive:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive=psutil.wait_procs(alive,timeout=10)
            for child in alive:
                child.kill()
            _, alive=psutil.wait_procs(alive,timeout=10)
            assert not [child for child in alive if child.status()!=psutil.STATUS_ZOMBIE]
        log.close()


if __name__=='__main__':
    main()
