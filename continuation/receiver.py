"""Receiver extension: immediate per-row durability, 25-row CSV snapshots."""
from __future__ import annotations
import argparse
import hmac
import json
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from zipfile import ZipFile
from filelock import FileLock
from streamlined import receiver as base
from streamlined.common import digest, read_csv, read_json
from .store import accept

ORIGINAL_BUNDLE = base.make_bundle

def extended_bundle(root):
    path=ORIGINAL_BUNDLE(root)
    with ZipFile(path,'a') as archive:
        for source in sorted((Path(root)/'continuation').glob('*.py')):
            archive.write(source, 'continuation/'+source.name)
    return path

def handler(root,token):
    root=Path(root)
    original=base.handler(root,token)
    prompt_hash={str(row['prompt_id']):row['prompt_sha256']
                 for row in read_csv(root/'data/prompts.csv')}
    class Handler(original):
        def handle_request(self):
            route=urlparse(self.path).path
            if not route.startswith('/continuation/'):
                return super().handle_request()
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send({'error':'Unauthorized'},401);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size>6*1024*1024:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(size)) if size else {}
                if body.get('_lease') and route != '/continuation/responses':
                    lease=body['_lease']
                    active=read_json(root/'runtime/heartbeats'/f'{digest(lease["run_id"])}.json')
                    if (active['owner']!=lease['owner'] or active['state']!='running'
                            or time.time()-active['received_at']>180):
                        self.send({'error':'Worker lease expired or replaced'},409);return
                query=parse_qs(urlparse(self.path).query)
                stream=body.get('stream') or query['stream'][0]
                if not stream.startswith('quantized/'):
                    raise ValueError('Only quantized result streams are accepted')
                folder=base.resolve(root,'results/'+stream)
                if route=='/continuation/responses':
                    directory=folder/'records'
                    rows=[read_json(p) for p in sorted(directory.glob('*.json'))] if directory.exists() else []
                    self.send({'rows':rows});return
                if route=='/continuation/snapshot':
                    self.send({'rows':accept(root,stream,[],final=True)});return
                if route not in ('/continuation/response','/continuation/response_batch'):
                    self.send({'error':'Unknown route'},404);return
                rows=[body['row']] if route.endswith('/response') else body['rows']
                for row in rows:
                    pid=str(row['prompt_id'])
                    if prompt_hash.get(pid)!=row.get('prompt_sha256'):
                        raise ValueError('Prompt ID/hash differs from frozen dataset')
                    if row.get('status')=='completed' and digest(row.get('response_raw',''))!=row.get('response_hash'):
                        raise ValueError('Response hash mismatch')
                # Acknowledgement occurs only after atomic records are durable.
                self.send({'rows':accept(root,stream,rows)})
            except Exception as error:
                self.send({'error':str(error)},400)
    return Handler

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    parser.add_argument('--port',type=int,default=8770)
    args=parser.parse_args();root=Path(args.root).resolve()
    token=base.token_file(root).read_text().strip()
    base.make_bundle=extended_bundle
    with FileLock(str(root/'runtime/receiver.lock'),timeout=0):
        print(f'Continuation receiver on 127.0.0.1:{args.port}; prior /response routes retained',flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),handler(root,token)).serve_forever()

if __name__=='__main__':main()
