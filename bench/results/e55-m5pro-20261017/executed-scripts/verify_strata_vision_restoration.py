"""Read-only post-trial checks; output only service and aggregate application state."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import psutil
root=Path('/Users/jakecantrell/inference-lab')
stage=root/'contributions/strata-vision-20261017'
photo=Path('/var/root/macdaddy_agent/AmazonPhotos')
state=json.loads((stage/'status.json').read_text())
assert state['status']=='complete' and state['restoration']=='passed'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
    return json.load(opener.open(url,timeout=10))
checks={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'health':get('http://127.0.0.1:12345/health')}
checks['context']=get('http://127.0.0.1:12345/props')['default_generation_settings']['n_ctx']
checks['launcher_unchanged']=hashlib.sha256((root/'service/start-inference.sh').read_bytes()).hexdigest()==state['launcher_sha256']
checks['plist_unchanged']=hashlib.sha256(Path('/Library/LaunchDaemons/com.cantrell.inference-lab.server.plist').read_bytes()).hexdigest()==state['plist_sha256']
viewer=json.loads((photo/'dashboard-process.json').read_text())
restore=json.loads((root/'contributions/strata-20261012/viewer-restore.json').read_text())
checks['viewer_pid']=viewer['pid']
checks['viewer_identity_matches']=restore['argv'][2] in subprocess.check_output(['ps','-p',str(viewer['pid']),'-o','args='],text=True)
checks['photo_action_busy']=get(viewer['url']+'/review/actions/status')['busy']
checks['aggregate_checkpoints_unchanged']=state['photo_before']==state['photo_restored']
checks['locks']={}
for name,path in [('scheduler',photo/'notifications/scheduled-scan.lock'),('actions',photo/'archive/.actions.lock'),('experiment',root/'contributions/strata-20261012/m5pro-runner.lock')]:
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks['locks'][name]='free'
    finally:os.close(fd)
checks['experimental_children']=[]
checks['production_logs_disabled']=False
for p in psutil.process_iter(['pid','cmdline','name']):
    cmd=p.info['cmdline'] or []
    if 'strata_mlx.serve' in cmd or 'serve.server' in cmd or (cmd and cmd[0].endswith('/strata-vision')):
        checks['experimental_children'].append(p.pid)
    if p.info['name']=='llama-server':checks['production_logs_disabled']='--log-disable' in cmd
assert checks['health']=={'status':'ok'} and checks['context']==131072
assert checks['launcher_unchanged'] and checks['plist_unchanged'] and checks['viewer_identity_matches']
assert not checks['photo_action_busy'] and checks['aggregate_checkpoints_unchanged']
assert not checks['experimental_children'] and checks['production_logs_disabled']
(stage/'restoration-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
