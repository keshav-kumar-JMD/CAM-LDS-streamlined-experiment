"""Colab notebook launcher for isolated, audibly monitored GGUF workers."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time
from streamlined.common import read_json
from streamlined.monitor import Heartbeat
from streamlined.transport import Mirror
from .protocol import condition_key, stream

def worker(root,model,representation,adapter=None,action='infer',pilot=None):
    root=Path(root)
    run=stream(model,representation,adapter)
    label=f'{model}_{condition_key(model,representation,adapter)}_{representation}_{action}'
    if pilot:label+=f'_pilot{pilot}'
    log=root/'telemetry/logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-u','-m','continuation.infer',action,'--root',str(root),
          '--model',model,'--representation',representation]
    if adapter:args.extend(['--adapter',adapter])
    if pilot:args.extend(['--pilot',str(pilot)])
    env=os.environ.copy();env['PYTHONPATH']=str(root);env['PYTHONFAULTHANDLER']='1'
    connection=read_json(root/'runtime/connection.json')
    mirror=Mirror(connection['url'],connection['token'])
    with Heartbeat(root,'launcher/'+run+'/'+action,mirror,stall_seconds=86400),log.open('a',encoding='utf-8') as output:
        process=subprocess.Popen(args,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
        print(f'Started {run}/{action}; PID={process.pid}; log={log}',flush=True)
        position=log.stat().st_size
        try:
            while process.poll() is None:
                time.sleep(2)
                with log.open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(position);chunk=reader.read();position=reader.tell()
                    if chunk:print(chunk,end='',flush=True)
            with log.open(encoding='utf-8',errors='replace') as reader:
                reader.seek(position);chunk=reader.read()
                if chunk:print(chunk,end='',flush=True)
            if process.returncode:raise RuntimeError(f'{run}/{action} exited {process.returncode}; see {log}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            try:mirror.put(root,log)
            except Exception as error:print('Worker log upload pending:',error,flush=True)
