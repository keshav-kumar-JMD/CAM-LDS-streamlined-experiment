"""Local-only receiver. Start watchdog in a separate process for audible alerts."""
from __future__ import annotations
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlparse,parse_qs
from filelock import FileLock
from .common import atomic,write_json,read_json,digest,secret,write_csv
from .transport import file_hash,make_bundle

def resolve(root,relative):
    root=Path(root).resolve(); path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root or any(x.startswith('.') for x in path.relative_to(root).parts):
        raise ValueError('Invalid artifact path')
    if path.relative_to(root).parts[0] not in ('results','telemetry','checkpoints','models','manifests'):
        raise ValueError('Artifact namespace not allowed')
    return path

def token_file(root):
    path=Path(root)/'runtime/receiver_token.txt'
    if not path.exists(): atomic(path,secrets.token_urlsafe(36))
    return path

def handler(root,token):
    root=Path(root)
    class Handler(BaseHTTPRequestHandler):
        def watchdog_alive(self):
            path=root/'runtime/watchdog_alive.json'
            return path.exists() and time.time()-read_json(path)['at']<30
        def log_message(self,*args): pass  # Do not log credentials or request bodies.
        def send(self,value,status=200,binary=False):
            body=value if binary else json.dumps(value).encode()
            self.send_response(status); self.send_header('Content-Type','application/octet-stream' if binary else 'application/json')
            self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self): self.handle_request()
        def do_POST(self): self.handle_request()
        def handle_request(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send({'error':'Unauthorized'},401); return
            try:
                parsed=urlparse(self.path); route=parsed.path; query=parse_qs(parsed.query)
                size=int(self.headers.get('Content-Length','0'))
                if size>6*1024*1024: raise ValueError('Request too large')
                obj=json.loads(self.rfile.read(size)) if size else {}
                if obj.get('_lease') and route!='/heartbeat':
                    lease=obj['_lease'];leasepath=root/'runtime/heartbeats'/f'{digest(lease["run_id"])}.json'
                    active=read_json(leasepath)
                    if active['owner']!=lease['owner'] or active['state']!='running' or time.time()-active['received_at']>180:
                        self.send({'error':'Worker lease expired or replaced'},409);return
                if route=='/health': self.send({'protocol':'camlds-streamlined-v1','watchdog_alive':self.watchdog_alive()}); return
                if route=='/bundle':
                    with FileLock(str(root/'runtime/bundle.lock')):content=make_bundle(root).read_bytes()
                    self.send(content,binary=True); return
                if route=='/credential/hf': self.send({'token':secret(root,'HF_Token.txt')}); return
                if route in ('/response','/responses'):
                    stream=obj.get('stream') if route=='/response' else query['stream'][0]
                    directory=resolve(root,'results/'+stream+'/records')
                    directory.mkdir(parents=True,exist_ok=True)
                    with FileLock(str(directory.parent/'records.lock')):
                        if route=='/response':
                            row=obj['row']; path=directory/(digest(row['record_id'])+'.json')
                            if path.exists():
                                previous=read_json(path)
                                if previous.get('status')=='completed' and (row.get('status')!='completed' or previous.get('response_hash')!=row.get('response_hash')):
                                    raise ValueError('Completed record conflicts with received response')
                                if previous.get('attempt_count')!=row.get('attempt_count'):
                                    write_json(directory.parent/'attempt_history'/f'{digest(row["record_id"])}_{previous.get("attempt_count",0)}.json',previous)
                            write_json(path,row)
                        rows=[read_json(p) for p in sorted(directory.glob('*.json'))]
                        write_csv(directory.parent/'responses.csv',rows)
                        self.send({'ok':True} if route=='/response' else {'rows':rows})
                    return
                if route in ('/artifact/info','/artifact/chunk'):
                    path=resolve(root,query['path'][0])
                    if route.endswith('info'):
                        self.send({'exists':path.is_file(),**({'size':path.stat().st_size,'sha256':file_hash(path)} if path.is_file() else {})}); return
                    with path.open('rb') as f:
                        f.seek(int(query['offset'][0])); self.send(f.read(4*1024*1024),binary=True)
                    return
                if route=='/heartbeat':
                    name=digest(obj['run_id']); path=root/'runtime/heartbeats'/f'{name}.json'
                    with FileLock(str(root/'runtime/leases.lock')):
                        if path.exists():
                            prev=read_json(path)
                            if prev.get('owner')!=obj['owner'] and prev.get('state')=='running' and time.time()-prev['received_at']<180:
                                self.send({'error':'Another live worker owns this run'},409); return
                        write_json(path,dict(obj,received_at=time.time()))
                    self.send({'ok':True,'watchdog_alive':self.watchdog_alive()}); return
                if route=='/upload/start':
                    path=resolve(root,obj['path']); upload_id=digest([obj['path'],obj['sha256']])
                    if path.is_file() and file_hash(path)==obj['sha256']:
                        self.send({'complete':True}); return
                    meta=root/'runtime/uploads'/f'{upload_id}.json'; part=meta.with_suffix('.part')
                    meta.parent.mkdir(parents=True,exist_ok=True)
                    with FileLock(str(meta)+'.lock'):
                        write_json(meta,obj)
                        self.send({'id':upload_id,'offset':part.stat().st_size if part.exists() else 0}); return
                if route.startswith('/upload/'):
                    uid=obj['id']
                    if len(uid)!=64 or any(x not in '0123456789abcdef' for x in uid): raise ValueError('Invalid upload ID')
                    meta=root/'runtime/uploads'/f'{uid}.json'; part=meta.with_suffix('.part')
                    with FileLock(str(meta)+'.lock'):
                        spec=read_json(meta); path=resolve(root,spec['path'])
                        if route=='/upload/chunk':
                            data=base64.b64decode(obj['data'],validate=True); offset=obj['offset']
                            actual=part.stat().st_size if part.exists() else 0
                            if offset<actual:
                                with part.open('rb') as f: f.seek(offset); existing=f.read(len(data))
                                if existing!=data: raise ValueError('Chunk retry differs')
                            elif offset==actual and actual+len(data)<=spec['size']:
                                with part.open('ab') as f:
                                    f.write(data); f.flush(); __import__('os').fsync(f.fileno())
                            else: raise ValueError('Invalid chunk offset')
                            self.send({'ok':True}); return
                        if route=='/upload/finish':
                            if path.is_file() and file_hash(path)==spec['sha256']:
                                self.send({'ok':True}); return
                            if part.stat().st_size!=spec['size'] or file_hash(part)!=spec['sha256']: raise ValueError('Upload checksum mismatch')
                            path.parent.mkdir(parents=True,exist_ok=True)
                            with FileLock(str(path)+'.lock'): __import__('os').replace(part,path)
                            self.send({'ok':True}); return
                self.send({'error':'Unknown route'},404)
            except Exception as error:
                self.send({'error':str(error)},400)
    return Handler

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--port',type=int,default=8770)
    a=p.parse_args(); root=Path(a.root).resolve(); token=token_file(root).read_text().strip()
    with FileLock(str(root/'runtime/receiver.lock'),timeout=0):
        print(f'Receiver listening on 127.0.0.1:{a.port}; token file: {token_file(root)}',flush=True)
        ThreadingHTTPServer(('127.0.0.1',a.port),handler(root,token)).serve_forever()
if __name__=='__main__': main()
