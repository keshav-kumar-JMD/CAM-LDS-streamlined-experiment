from pathlib import Path
import os
import subprocess
import sys
import time
from .common import *
from .transport import Mirror
from .monitor import Heartbeat

def local_root():
    cwd=Path.cwd().resolve()
    for root in (cwd,*cwd.parents):
        if (root/'streamlined/common.py').exists():return root
        if (root/'streamlined_experiment/streamlined/common.py').exists():return root/'streamlined_experiment'
    raise FileNotFoundError('Open this notebook within streamlined_experiment')

def watchdog_ready(root):
    path=Path(root)/'runtime/watchdog_alive.json'
    if not path.exists() or time.time()-read_json(path)['at']>20:
        raise RuntimeError('Start the independent local watchdog using the commands above this cell')

def connect(root,url,token):
    m=Mirror(url,token)
    if m.call('/health')['protocol']!='camlds-streamlined-v1':raise ValueError('Use streamlined receiver port 8770')
    write_json(Path(root)/'runtime/connection.json',{'url':url,'token':token})
    os.chmod(Path(root)/'runtime/connection.json',0o600)
    return m

def worker(root,model,action,condition,split='all'):
    root=Path(root);run=f'{model}/{action}/{condition}/{split}'
    path=root/'telemetry/logs'/f'{model}_{action}_{condition}_{split}.log';path.parent.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-u','-m','streamlined.gpu',action,'--root',str(root),'--model',model,'--condition',condition,'--split',split]
    env=os.environ.copy();env['PYTHONPATH']=str(root);env['PYTHONFAULTHANDLER']='1'
    connection=read_json(root/'runtime/connection.json')
    supervisor=Mirror(connection['url'],connection['token'])
    with Heartbeat(root,'launcher/'+run,supervisor,stall_seconds=86400),path.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(f'Started {run}; PID={process.pid}; log={path}',flush=True)
        position=path.stat().st_size
        try:
            while process.poll() is None:
                time.sleep(2)
                with path.open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(position);chunk=reader.read();position=reader.tell()
                    if chunk:print(chunk,end='',flush=True)
            if process.returncode:
                raise RuntimeError(f'Worker exited {process.returncode}; rerun connection and this stage. See {path}')
        except BaseException as error:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            # Report immediately when the kernel survives. Heartbeat expiry also
            # catches kernel/runtime disappearance when this handler cannot run.
            try:
                logical=f'{model}/train/{condition}' if action=='train' else f'{model}/{condition}/{split}'
                state=root/'runtime/heartbeats'/f'{digest(logical)}.json'
                if state.exists():
                    data=read_json(state);data.update(state='failed',stage=str(error)[:300])
                    c=read_json(root/'runtime/connection.json')
                    Mirror(c['url'],c['token']).call('/heartbeat',data)
            except Exception:pass
            raise
        finally:
            try:
                c=read_json(root/'runtime/connection.json');Mirror(c['url'],c['token']).put(root,path)
            except Exception:pass

def billing(root,session,start_units,end_units,rate_per_hour=None):
    row=dict(record_id=session,session=session,start_units=start_units,end_units=end_units,
             compute_units_consumed=start_units-end_units if start_units is not None and end_units is not None else None,
             rate_per_hour=rate_per_hour,scope='manual Colab account balance; concurrent sessions may contribute')
    upsert_csv(Path(root)/'telemetry/colab_billing.csv',row)
    write_json(Path(root)/'telemetry/billing'/f'{digest(session)}.json',row)
    return row
