"""Pinned, synthetic Mac experiment with workload gate and restoration in finally.

Waits for the separately staged public packs; no credentials or private photo inputs.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/Users/jakecantrell/inference-lab')
STAGE = ROOT/'contributions/strata-20261012'
SOURCE = STAGE/'source-8a5aeaa'
PYTHON = ROOT/'runtimes/strata-mlx/.venv/bin/python'
REVISION = '8a5aeaafcb6fe4eefed296e8424e9ec3817250c0'
RESUME = Path(os.environ['STRATA_RESUME_RESULTS']) if os.environ.get('STRATA_RESUME_RESULTS') else None
PLIST = Path('/Library/LaunchDaemons/com.cantrell.inference-lab.server.plist')
JOB = 'system/com.cantrell.inference-lab.server'
PHOTO = Path('/var/root/macdaddy_agent/AmazonPhotos')
PHOTO_DATA = PHOTO/'benchmark-latest100'
RUN = STAGE/'results'/time.strftime(os.environ.get('STRATA_RUN_PREFIX','cycle')+'-%Y%m%dT%H%M%SZ', time.gmtime())
RUN.mkdir(parents=True, exist_ok=False)
STATE_PATH = STAGE/'cycle-status.json'
state = {'status': 'waiting-downloads', 'revision': REVISION, 'results': str(RUN), 'steps': []}
env = dict(os.environ)
env.pop('MLX_ENABLE_TF32', None)
env['PYTHONPATH'] = str(SOURCE)
env['PATH'] = str(ROOT/'runtimes/llama-unsloth-b11505-mix-5209c9f/llama-b11505-mix-5209c9f')+os.pathsep+env.get('PATH', '')
if RESUME:
    env['PATH'] = str(STAGE/'tokenizer-adapter')+os.pathsep+env['PATH']
    state['prior_results'] = str(RESUME)
    state['tokenizer_adapter'] = 'Vocabulary/BPE-verified Python adapter for patched llama-tokenize context bug'
viewer = json.loads((STAGE/'viewer-restore.json').read_text())
viewer_paused = service_stopped = False
child = None

def save(phase=None):
    if phase:
        state['status'] = phase
    state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    STATE_PATH.write_text(json.dumps(state, indent=2)+'\n')
    print(state['updated_utc']+' '+state['status'], flush=True)

def get(path, timeout=10):
    with urllib.request.urlopen('http://127.0.0.1:12345'+path, timeout=timeout) as response:
        return json.load(response)

def command(args, label, cwd=SOURCE, timeout=2400, require=True, process_env=None):
    global child
    row = {'label': label, 'command': list(map(str,args)), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    state['steps'].append(row)
    save(label)
    with (RUN/(label+'.log')).open('w') as log:
        child = subprocess.Popen(list(map(str,args)), cwd=cwd, env=process_env or env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = child.pid
        save()
        try:
            row['exit'] = child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            child = None
            save()
    if require and row['exit']:
        raise RuntimeError('Step failed: '+label)
    return row['exit']

def photo_states():
    with sqlite3.connect('file:'+str(PHOTO_DATA/'series-lane.sqlite')+'?mode=ro', uri=True) as db:
        row = db.execute('SELECT payload FROM latest_run WHERE id=1').fetchone()
    primary = json.loads(row[0]) if row else {'state': 'idle'}
    adult_path = PHOTO_DATA/'adult-run.json'
    adult = json.loads(adult_path.read_text()) if adult_path.exists() else {'state': 'idle'}
    keys = ('state','processed','remaining','planned','failed','cached','interrupted')
    return {label: {k:d[k] for k in keys if k in d} for label,d in [('primary',primary),('adult',adult)]}

def wait_photo_and_pause():
    global viewer_paused
    save('waiting-photo-boundary')
    deadline = time.monotonic()+1800
    while True:
        lanes = photo_states()
        processes = subprocess.check_output(['ps','-axo','pid=,args='],text=True).splitlines()
        other_clients = [line for line in processes if ('amazonphotos' in line or 'AmazonPhotos' in line)
                         and not line.lstrip().startswith(str(viewer['pid'])+' ')
                         and not line.lstrip().startswith(str(os.getpid())+' ')]
        if all(x.get('state') != 'running' for x in lanes.values()) and not other_clients:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Photo workload did not reach a clean boundary')
        time.sleep(15)
    state['photo_before'] = lanes
    actual = subprocess.check_output(['ps','-p',str(viewer['pid']),'-o','args='],text=True).strip()
    if viewer['argv'][2] not in actual:
        raise RuntimeError('Viewer identity changed; refusing to stop an unknown process')
    # No automatic resume occurs on viewer restart. Preserve its durable partial/completed states.
    os.kill(viewer['pid'],signal.SIGTERM)
    viewer_paused = True
    deadline = time.monotonic()+90
    while subprocess.run(['kill','-0',str(viewer['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0:
        if time.monotonic() > deadline:
            raise RuntimeError('Viewer did not exit cleanly; test postponed')
        time.sleep(2)
    state['photo_after_pause'] = photo_states()
    # Checkpoint could have changed between observation and signal; never load after a running snapshot.
    if any(x.get('state') == 'running' for x in state['photo_after_pause'].values()):
        raise RuntimeError('Photo work was still running at pause')
    for _ in range(3):
        if any(x.get('is_processing') for x in get('/slots')):
            raise RuntimeError('Inference has an outstanding request')
        time.sleep(2)
    save('photo-paused-and-drained')

def snapshot(label):
    data = {}
    for key,args in [('vm',['vm_stat']),('swap',['sysctl','vm.swapusage']),('wired',['sysctl','iogpu.wired_limit_mb'])]:
        data[key] = subprocess.check_output(args,text=True).strip()
    (RUN/(label+'.json')).write_text(json.dumps(data,indent=2)+'\n')

def smoke(image=False):
    content = 'Reply exactly READY.'
    if image:
        content = [{'type':'text','text':'Read the printed text in this synthetic image. Return only that text.'},
                   {'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                    base64.b64encode((STAGE/'fixtures/restore-ocr.png').read_bytes()).decode()}}]
    payload = {'model':'flash-next-uncensored','messages':[{'role':'user','content':content}],
               'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    request = urllib.request.Request('http://127.0.0.1:12345/v1/chat/completions',json.dumps(payload).encode(),
                                     {'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:
        result = json.load(response)
    answer = result['choices'][0]['message']['content'].strip()
    okay = '731' in answer and 'STRATA' in answer.upper() if image else answer == 'READY'
    if not okay:
        raise RuntimeError('Restoration generation check failed')
    return {'pass':okay,'synthetic_output_sha256':hashlib.sha256(answer.encode()).hexdigest()}

def stop_service():
    global service_stopped
    command(['launchctl','print',JOB],'service-before',cwd=STAGE,timeout=15)
    state['launcher_sha256'] = hashlib.sha256((ROOT/'service/start-inference.sh').read_bytes()).hexdigest()
    state['plist_sha256'] = hashlib.sha256(PLIST.read_bytes()).hexdigest()
    service_stopped = True  # Even a partial bootout must reach the restoration guard.
    command(['launchctl','bootout',JOB],'service-stop',cwd=STAGE,timeout=60)
    deadline = time.monotonic()+120
    while True:
        alive = subprocess.run(['pgrep','-x','llama-server'],capture_output=True).returncode == 0
        if not alive:
            try:
                get('/health',2)
            except Exception:
                break
        if time.monotonic() > deadline:
            raise RuntimeError('Resident model did not stop')
        time.sleep(2)

def restore():
    global service_stopped, viewer_paused
    save('restoring')
    if service_stopped:
        loaded = subprocess.run(['launchctl','print',JOB],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
        if not loaded:
            subprocess.run(['launchctl','bootstrap','system',str(PLIST)],check=True,timeout=60)
        deadline = time.monotonic()+240
        while True:
            try:
                if get('/health',5).get('status') == 'ok':
                    break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Flash did not regain health')
            time.sleep(3)
        state['restored_text'] = smoke()
        state['restored_image'] = smoke(True)
        if hashlib.sha256((ROOT/'service/start-inference.sh').read_bytes()).hexdigest() != state['launcher_sha256']:
            raise RuntimeError('Serving launcher changed during the trial')
        service_stopped = False
    if viewer_paused:
        # If clean shutdown failed and the original process still exists, do not start a duplicate.
        alive = subprocess.run(['kill','-0',str(viewer['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
        if not alive:
            p = subprocess.Popen(viewer['argv'],cwd=viewer['cwd'],stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            state['restored_viewer_pid'] = p.pid
            metadata = PHOTO/'dashboard-process.json'
            d = json.loads(metadata.read_text());d['pid']=p.pid;metadata.write_text(json.dumps(d)+'\n')
            time.sleep(5)
            if p.poll() is not None:
                raise RuntimeError('Photo viewer failed to restart')
        state['photo_restored'] = photo_states()
        viewer_paused = False
    snapshot('after-restoration')

def interrupted():
    raise InterruptedError('Experiment interrupted; restoring serving state')

def main():
    signal.signal(signal.SIGTERM,lambda *_: interrupted())
    signal.signal(signal.SIGINT,lambda *_: interrupted())
    save()
    failure = None
    try:
        while True:
            download = json.loads((STAGE/'download-status.json').read_text())
            if download['status'] == 'failed':
                raise RuntimeError('Pinned model staging failed')
            if download['status'] == 'complete':
                break
            time.sleep(15)
        state['models'] = download['verified']
        venv = SOURCE/'.venv'
        if not venv.exists():
            venv.symlink_to(ROOT/'runtimes/strata-mlx/.venv',target_is_directory=True)
        fixtures = json.loads((STAGE/'fixtures/synthetic-v1.json').read_text())
        bounded = dict(fixtures);bounded['cases']=[x for x in fixtures['cases'] if x['kind']=='short' or x['name']=='context-10000']
        fixture_path = RUN/'synthetic-10k.json';fixture_path.write_text(json.dumps(bounded,indent=2)+'\n')
        wait_photo_and_pause()
        snapshot('before-control')
        if not RESUME:
            command([PYTHON,STAGE/'scripts/strata_bench.py','run','--backend','http','--fixtures',fixture_path,
                     '--out',RUN/'flash-control','--label','flash-control','--revision','b11505-mix-5209c9f',
                     '--repeats','3','--maintenance-ready'],'flash-control',cwd=STAGE,timeout=2400)
        stop_service()
        snapshot('after-unload')
        for precision in ([] if RESUME else ['unset','0']):
            numerical_env=dict(env)
            if precision=='0': numerical_env['MLX_ENABLE_TF32']='0'
            command([PYTHON,'-m','pytest','-q'],'numerical-'+precision,timeout=900,require=False,process_env=numerical_env)
        model_root=ROOT/'models/standard-gsq/ed59f92082b1'
        models=[('IQ3_S',model_root/'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),
                ('Q2_0',model_root/'Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'),
                ('IQ3_XXS',ROOT/'models/uncensored-gsq-iq3/eec4e10a2c29/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf')]
        draft=ROOT/'models/strata-mlx-mtp'
        tokenizer=ROOT/'models/uncensored-affine38/2cab8310cb9b/tokenizer.json'
        for name,model in models:
            for budget in [0,4096,8192]:
                label=name+'-cache-'+str(budget)
                if RESUME:
                    report=json.loads((RESUME/label/'report.json').read_text())
                    assert report['status']=='complete'
                    assert report['revision']==REVISION
                    if not all(r['correct'] for r in report['rows'] if r['case']=='context-10000'):
                        state.setdefault('retention_failures',[]).append(name)
                    continue
                command([PYTHON,'-m','strata_mlx.serve','--check','--native',model,'--conversation-cache-mib',str(budget)],label+'-check',timeout=120)
                native=[str(PYTHON),'-m','strata_mlx.serve','--serve','--native',str(model),'--mtp',str(draft),
                        '--max-context','73728','--dtype','float16','--expert-profile','off','--conversation-cache-mib',str(budget)]
                command([PYTHON,STAGE/'scripts/strata_bench.py','run','--backend','native','--fixtures',fixture_path,
                         '--out',RUN/label,'--label',label,'--revision',REVISION,'--repeats','3','--maintenance-ready',
                         '--cwd',SOURCE,'--tokenizer',tokenizer,'--command-json',json.dumps(native)],label,timeout=3600)
                report=json.loads((RUN/label/'report.json').read_text())
                if not all(r['correct'] for r in report['rows']):
                    state.setdefault('correctness_failures',[]).append(label)
                snapshot(label+'-after')
        # Maintainer's unmodified scripts: targeted source behavior and protocol comparison.
        iq3=models[0][1]
        command([PYTHON,'bench/compare_servers.py','--model',iq3,'--mtp',draft,'--only','strata-mlx default',
                 '--log-dir',RUN,'--engine-args','--expert-profile off','--json',RUN/'upstream-compare.json'],
                'upstream-compare',timeout=3600)
        command([PYTHON,'bench/prompt_by_layer.py','--model',iq3,'--mtp',draft,'--machine-memory','64',
                 '--json',RUN/'upstream-layer.json'],'upstream-layer',timeout=5400)
        # Follow the bounded 10K gate; expand only when that pack's three arms all passed.
        for name,model in models:
            if (RESUME and name in state.get('retention_failures',[])) or (not RESUME and any(x.startswith(name+'-cache-') for x in state.get('correctness_failures',[]))):
                continue
            longer=dict(fixtures);longer['cases']=[x for x in fixtures['cases'] if x['name'] in ('context-8192','context-32768')]
            longer_path=RUN/'synthetic-long.json';longer_path.write_text(json.dumps(longer,indent=2)+'\n')
            label=name+'-long-cache4096'
            native=[str(PYTHON),'-m','strata_mlx.serve','--serve','--native',str(model),'--mtp',str(draft),
                    '--max-context','73728','--dtype','float16','--expert-profile','off','--conversation-cache-mib','4096']
            command([PYTHON,STAGE/'scripts/strata_bench.py','run','--backend','native','--fixtures',longer_path,
                     '--out',RUN/label,'--label',label,'--revision',REVISION,'--repeats','3','--maintenance-ready',
                     '--cwd',SOURCE,'--tokenizer',tokenizer,'--command-json',json.dumps(native)],label,timeout=3600)
        state['benchmarks_finished']=True
    except BaseException as error:
        failure=error
        state['failure_type']=type(error).__name__
        state['failure_step']=state['status']
    finally:
        try:
            restore()
            state['restoration']='passed'
        except BaseException as error:
            state['restoration']='failed'
            state['restoration_error_type']=type(error).__name__
            failure=failure or error
        save('failed' if failure else 'complete')
        (RUN/'cycle-report.json').write_text(json.dumps(state,indent=2)+'\n')
    if failure:
        sys.exit(1)


if __name__ == '__main__':
    main()
