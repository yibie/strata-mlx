"""Exclusive synthetic latency screen, bracketed production controls and recovery."""
import os
os.environ['STRATA_RUN_PREFIX']='vision'
os.environ.pop('MLX_ENABLE_TF32',None)
os.environ['MTL_DEBUG_LAYER']='0'
import fcntl
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path('/Users/jakecantrell/inference-lab')
OLD=ROOT/'contributions/strata-20261012'
STAGE=ROOT/'contributions/strata-vision-20261017'
sys.path.insert(0,str(OLD/'scripts'))
import run_strata_cycle as guard
import strata_bench as bench
lock=(OLD/'m5pro-runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
scan_fd=os.open(guard.PHOTO/'notifications/scheduled-scan.lock',os.O_RDWR|os.O_NOFOLLOW)
fcntl.flock(scan_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
guard.STATE_PATH=STAGE/'status.json'
guard.state={'status':'preflight','steps':[],'results':str(guard.RUN),'source':'vision-zmuonvlz-on-local-b4b1a2ce',
             'scheduler_lock':'held through restoration','protocol':'resident-serving-v1'}
fixtures=[]
source=STAGE/'source'
failure=None

def http_control(label):
    guard.save(label)
    for repeat in range(2):
        for case in fixtures:
            body={'prompt':case['rendered'],'n_predict':case['max_tokens'],'temperature':0,'seed':42,
                  'cache_prompt':False,'stream':True}
            req=urllib.request.Request('http://127.0.0.1:12345/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
            text='';ttft=None;final=None;start=time.perf_counter()
            with urllib.request.urlopen(req,timeout=300) as response:
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data: ') or line=='data: [DONE]':continue
                    item=json.loads(line[6:]);content=item.get('content','')
                    if content and ttft is None:ttft=time.perf_counter()-start
                    text+=content
                    if item.get('stop'):final=item
            if final is None:raise RuntimeError('Control stream incomplete')
            row={'label':label,'case':case['name'],'round':repeat,'text':text,'ttft_s':ttft,
                 'wall_s':time.perf_counter()-start,'timings':final.get('timings'),
                 'stop_type':final.get('stop_type'),'stopped_limit':final.get('stopped_limit'),
                 'tokens_predicted':final.get('tokens_predicted'),'tokens_evaluated':final.get('tokens_evaluated')}
            with (guard.RUN/'llama-control.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            guard.save(label+'-'+case['name']+'-'+str(repeat))

def swapout_bytes():
    text=subprocess.check_output(['vm_stat'],text=True)
    return int(re.search(r'^Swapouts:\s+(\d+)\.',text,re.M)[1])*int(re.search(r'page size of (\d+) bytes',text)[1])

original_pause=guard.wait_photo_and_pause

def checked_pause():
    dashboard=json.loads((guard.PHOTO/'dashboard-process.json').read_text())['url']
    deadline=time.monotonic()+1800
    while True:
        def read_status(path):
            return json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(dashboard+path,timeout=10))
        actions=read_status('/review/actions/status')
        if not isinstance(actions.get('busy'),bool):raise RuntimeError('Photo action status schema changed')
        if not actions['busy'] and all(x.get('state')!='running' for x in guard.photo_states().values()):
            break
        if time.monotonic()>deadline:raise RuntimeError('Photo action/lane did not become idle')
        time.sleep(10)
    fd=os.open(guard.PHOTO/'archive/.actions.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original_pause()
    finally:os.close(fd)

guard.wait_photo_and_pause=checked_pause

signal.signal(signal.SIGTERM,lambda *_:guard.interrupted())
signal.signal(signal.SIGINT,lambda *_:guard.interrupted())
try:
    pid=json.loads((guard.PHOTO/'dashboard-process.json').read_text())['pid']
    actual=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True).strip()
    if guard.viewer['argv'][2] not in actual:raise RuntimeError('Viewer identity changed')
    guard.viewer['pid']=pid
    guard.wait_photo_and_pause();guard.snapshot('before')
    guard.state['preflight_health']=guard.get('/health')
    guard.state['preflight_props']=guard.get('/props').get('default_generation_settings',{}).get('n_ctx')
    guard.stop_service()
    env=dict(os.environ);env.update(PYTHONPATH=str(source),PATH=str(OLD/'review-tools')+os.pathsep+guard.env['PATH'])
    guard.env=env
    guard.command([guard.PYTHON,'-m','pytest','-q'],'upstream-contribution-tests',cwd=STAGE/'upstream-source',timeout=600)
    model=ROOT/'models/uncensored-gsq-iq3/eec4e10a2c29/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf'
    before=swapout_bytes();stop=threading.Event()
    def watch():
        while not stop.wait(2):
            if swapout_bytes()-before>512*1024**2:
                guard.state['memory_guard_abort']=True
                if guard.child:os.killpg(guard.child.pid,signal.SIGTERM)
                return
    t=threading.Thread(target=watch,daemon=True);t.start()
    try:
        with bench.MemorySampler(guard.RUN/'screen-memory.jsonl'):
            official=ROOT/'contributions/strata-official-296b59d'
            cfg={'exe':str(STAGE/'mlx-vision'),'log':str(guard.RUN/'synthetic-engine.log'),
                 'args':['--native',str(model),'--mtp',str(ROOT/'models/strata-mlx-mtp'),
                         '--expert-memory','resident','--vision','--max-context','16384',
                         '--fusion-limit','6','--conversation-cache-mib','1024',
                         '--conversation-cache-min-tokens','0','--expert-profile','off','--keep-awake','2'],
                 'cwd':str(source),'tokenizer':str(official/'comparison-pack/tokenizer'),
                 'model_name':'flash-next-uncensored','env':{'MTL_DEBUG_LAYER':'0'},
                 'vision':{'exe':str(official/'build-vision/bin/strata-vision'),
                           'gpu':True,'max_tokens':1024,'model':str(model),
                           'mmproj':str(ROOT/'models/uncensored-flash-vision/eec4e10a2c29/mmproj-Qwen3.8-Flash-Next-BF16.gguf')}}
            config=guard.RUN/'api-config.json';config.write_text(json.dumps(cfg,indent=2))
            guard.command([official/'.venv/bin/python', STAGE/'probe_vision_api.py', '--config',config,
                           '--out',guard.RUN/'api'], 'vision-api', cwd=official,timeout=1200)
    finally:stop.set();t.join(timeout=5)
except BaseException as error:
    failure=error;guard.state.update(failure_type=type(error).__name__,failure_message=str(error))
finally:
    try:
        guard.restore();guard.state['restoration']='passed'
        if failure is None:
            # Viewer must remain paused during control generation; restore() restarts it.
            # Re-pause its verified fresh identity under the same scheduler lock.
            pid=json.loads((guard.PHOTO/'dashboard-process.json').read_text())['pid']
            guard.viewer['pid']=pid
            guard.wait_photo_and_pause()
            guard.smoke();guard.smoke(True)
            guard.restore()
    except BaseException as error:
        failure=failure or error;guard.state.update(restoration='failed',restoration_error=str(error))
        try:guard.restore();guard.state['recovery_retry']='passed'
        except BaseException as retry:guard.state['recovery_retry_error']=str(retry)
    if guard.state.get('photo_restored') != guard.state.get('photo_before'):
        failure=failure or RuntimeError('Aggregate photo checkpoints changed')
    if guard.state.get('plist_sha256') and hashlib.sha256(guard.PLIST.read_bytes()).hexdigest()!=guard.state['plist_sha256']:
        failure=failure or RuntimeError('Production plist changed')
    guard.save('failed' if failure else 'complete')
    (guard.RUN/'report.json').write_text(json.dumps(guard.state,indent=2)+'\n')
if failure:raise SystemExit(1)
