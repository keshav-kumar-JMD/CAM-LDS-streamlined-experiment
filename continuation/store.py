"""Durable row acknowledgements with infrequent CSV materialization."""
from __future__ import annotations
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from urllib.parse import quote
from filelock import FileLock
from streamlined.common import digest, read_json, write_json, write_csv
from .protocol import SNAPSHOT_EVERY, UPLOAD_BACKPRESSURE, UPLOAD_BATCH_SIZE

DONE = {'completed', 'context_unsupported'}

def record_path(root, stream, record_id):
    return Path(root)/'results'/stream/'records'/(digest(record_id)+'.json')

def export(root, stream):
    folder = Path(root)/'results'/stream
    records = [read_json(p) for p in sorted((folder/'records').glob('*.json'))]
    records.sort(key=lambda r: (int(r['prompt_id']), r.get('record_id','')))
    write_csv(folder/'responses.csv', records)
    return len(records)

def accept(root, stream, rows, final=False):
    if not stream.startswith('quantized/') or '..' in stream.split('/'):
        raise ValueError('Invalid quantized stream')
    folder = Path(root)/'results'/stream
    directory = folder/'records'; directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder/'records.lock')):
        changed=0
        for row in rows:
            if row.get('record_id') is None or row.get('prompt_id') is None:
                raise ValueError('Missing row identity')
            target = record_path(root, stream, row['record_id'])
            if target.exists():
                previous = read_json(target)
                if previous.get('status') in DONE:
                    if row.get('status') != previous['status'] or row.get('response_hash') != previous.get('response_hash'):
                        raise ValueError('Completed response conflicts with saved checkpoint')
                    continue
                if previous==row:continue
            write_json(target, row)
            changed+=1
        count = len(list(directory.glob('*.json')))
        marker = folder/'exported_count.json'
        state=read_json(marker) if marker.exists() else {}
        pending=int(state.get('pending_mutations',0))+changed
        if final or pending >= SNAPSHOT_EVERY:
            export(root,stream)
            pending=0
        if changed or final:
            write_json(marker, {'count':count,'pending_mutations':pending})
    return count

class SyncStore:
    def __init__(self, root, stream, mirror):
        self.root=Path(root); self.stream=stream; self.mirror=mirror
        self.folder=self.root/'results'/stream
        self.folder.mkdir(parents=True,exist_ok=True)
        local={r['record_id']:r for p in (self.folder/'records').glob('*.json') if (r:=read_json(p))}
        remote=mirror.call('/continuation/responses?stream='+quote(stream,safe=''))['rows']
        self.rows={r['record_id']:r for r in remote}
        pending=[]
        for rid,row in local.items():
            old=self.rows.get(rid)
            if old and old.get('status') in DONE:
                if row.get('status') in DONE and old.get('response_hash')!=row.get('response_hash'):
                    raise ValueError('Local and laptop responses conflict')
            elif row.get('status') in DONE:
                pending.append(row)
        if pending:
            # Batch transfer is for replay of already saved, locally spooled rows.
            for start in range(0,len(pending),UPLOAD_BATCH_SIZE):
                chunk=pending[start:start+UPLOAD_BATCH_SIZE]
                mirror.call('/continuation/response_batch',{'stream':stream,'rows':chunk})
                self.rows.update({r['record_id']:r for r in chunk})
        for row in self.rows.values():
            target=record_path(root,stream,row['record_id'])
            if not target.exists():write_json(target,row)
        self.queue=Queue()
        self.condition=threading.Condition()
        self.outstanding=0
        self.upload_seconds=0.0
        self.upload_error=None
        self.closing=False
        self.thread=threading.Thread(target=self._upload_loop,daemon=True,
                                     name='checkpoint-'+digest(stream)[:8])
        self.thread.start()

    def _upload_loop(self):
        while True:
            try:first=self.queue.get(timeout=.2)
            except Empty:
                with self.condition:
                    if self.closing and self.outstanding==0:return
                continue
            batch=[first]
            while len(batch)<UPLOAD_BATCH_SIZE:
                try:batch.append(self.queue.get(timeout=.05))
                except Empty:break
            delay=1
            while True:
                try:
                    started=time.perf_counter()
                    self.mirror.call('/continuation/response_batch',
                                     {'stream':self.stream,'rows':batch})
                    elapsed=time.perf_counter()-started
                    with self.condition:
                        self.upload_seconds+=elapsed
                        self.rows.update({r['record_id']:r for r in batch})
                        self.outstanding-=len(batch)
                        self.condition.notify_all()
                    for _ in batch:self.queue.task_done()
                    break
                except Exception as error:
                    with self.condition:self.upload_error=error
                    time.sleep(delay);delay=min(delay*2,30)

    def flush(self):
        with self.condition:
            while self.outstanding:
                self.condition.wait(timeout=1)
        return self.upload_seconds

    def done(self, rid):return self.rows.get(rid,{}).get('status') in DONE

    def put(self, row):
        import time
        rid=row['record_id']; previous=self.rows.get(rid)
        if previous and previous.get('status') in DONE:
            if previous.get('response_hash')!=row.get('response_hash'):
                raise ValueError('Completed response changed')
            return {'local_seconds':0.0,'receiver_seconds':0.0}
        started=time.perf_counter()
        write_json(record_path(self.root,self.stream,rid),row)
        local_seconds=time.perf_counter()-started
        self.rows[rid]=row
        with self.condition:
            self.outstanding+=1
            self.queue.put(row)
            while self.outstanding>=UPLOAD_BACKPRESSURE:
                self.condition.wait(timeout=1)
        return {'local_seconds':local_seconds,'receiver_seconds':0.0}

    def snapshot(self):
        self.flush()
        self.mirror.call('/continuation/snapshot',{'stream':self.stream})
        return export(self.root,self.stream)

    def close(self):
        self.flush()
        with self.condition:
            self.closing=True
            self.condition.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('Checkpoint uploader did not stop')

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
