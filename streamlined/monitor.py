"""Heartbeat producer and independent audible watchdog (Windows)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import threading
import time
import uuid
from filelock import FileLock
from .common import write_json,read_json,digest

class Heartbeat:
    def __init__(self,root,run_id,mirror=None,stall_seconds=1800):
        self.root=Path(root); self.run_id=run_id; self.mirror=mirror
        self.owner=uuid.uuid4().hex; self.state='running'; self.stage='starting'
        self.last_progress=time.time();self.progress='';self.stall_seconds=stall_seconds
        self.stop=threading.Event(); self.error=None;self.thread=None;self.final_state='completed'
    def emit(self):
        data=dict(run_id=self.run_id,owner=self.owner,pid=os.getpid(),state=self.state,
            stage=self.stage,progress=self.progress,last_progress=self.last_progress,
            stall_seconds=self.stall_seconds,received_at=time.time())
        write_json(self.root/'runtime/heartbeats'/f'{digest(self.run_id)}.json',data)
        if self.mirror:
            response=self.mirror.call('/heartbeat',data)
            if not response.get('watchdog_alive'):raise RuntimeError('Local watchdog is not running; restart it before model work')
    def loop(self):
        while not self.stop.wait(30):
            try:self.emit();self.error=None
            except Exception as error:self.error=error
    def __enter__(self):
        self.emit()
        if self.mirror:self.mirror.lease={'run_id':self.run_id,'owner':self.owner}
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def update(self,progress,stage=None):
        self.progress=str(progress);self.last_progress=time.time()
        if stage:self.stage=stage
        self.emit()
    def check(self):
        if self.error: raise OSError('Heartbeat/checkpoint receiver unavailable') from self.error
    def __exit__(self,kind,value,tb):
        self.stop.set()
        if self.thread:self.thread.join(timeout=50)
        self.state='failed' if kind else self.final_state
        self.stage=str(value)[:300] if kind else ('finished' if self.final_state=='completed' else 'poll window ended; rerun polling')
        try:self.emit()
        except Exception:pass # Watchdog detects the stale heartbeat if delivery is impossible.

def sound(urgent=True):
    try:
        import winsound
        if urgent:
            for frequency in (880,660,880):winsound.Beep(frequency,350)
        else:winsound.MessageBeep(winsound.MB_OK)
        return 'Windows winsound'
    except (ImportError,RuntimeError):
        print('Native audio unavailable; attempting terminal bell.\a',flush=True)
        return 'terminal bell fallback'

def reasons(root,now=None):
    now=time.time() if now is None else now;alerts=[]
    for p in (Path(root)/'runtime/heartbeats').glob('*.json'):
        try:r=read_json(p)
        except (OSError,ValueError):continue
        reason=None
        if r['state'] in ('failed','attention'):reason='attention required: '+r.get('stage','')
        elif r['state']=='running':
            if now-r['received_at']>150:reason='heartbeat lost: runtime, process, network or tunnel interruption'
            elif now-r['last_progress']>r.get('stall_seconds',1800):reason='progress stalled'
        if reason:alerts.append((r['run_id'],reason,r['owner']))
    return alerts

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--test',action='store_true')
    p.add_argument('--ack',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if a.test:
        mechanism=sound();print('Alarm test:',mechanism+'. Verify it was audible.');return
    if a.ack:
        write_json(root/'runtime/ack.json',{'at':time.time()});return
    (root/'runtime').mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'runtime/watchdog.lock'),timeout=0):
        print('Watchdog armed. Keep this process running and Windows awake/unmuted.',flush=True)
        acknowledged=set();previous=set();completed=set();last_alarm=0;ack_time=0
        while True:
            alerts=set(reasons(root))
            ack=root/'runtime/ack.json'
            if ack.exists():
                current=read_json(ack)['at']
                if current>ack_time:acknowledged.update(alerts);ack_time=current
            acknowledged.intersection_update(alerts)
            for alert in alerts-previous:print('ALERT:',alert[0],alert[1],flush=True)
            if alerts-acknowledged and time.time()-last_alarm>15:sound();last_alarm=time.time()
            for path in (root/'runtime/heartbeats').glob('*.json'):
                try:
                    r=read_json(path);identity=(r['run_id'],r['owner'])
                    if r['state']=='completed' and identity not in completed:
                        completed.add(identity)
                        if time.time()-r['received_at']<90:sound(False);print('Completed:',r['run_id'],flush=True)
                except (ValueError,OSError):pass
            write_json(root/'runtime/watchdog_alive.json',{'at':time.time()})
            previous=alerts;time.sleep(5)
if __name__=='__main__':main()
