"""Authenticated chunked transfer; only acknowledged artifacts are durable."""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
import zipfile
from .common import atomic, digest, read_json

class Mirror:
    def __init__(self,url,token):
        from urllib.parse import urlparse
        p=urlparse(url)
        if p.scheme!='https' and not (p.scheme=='http' and p.hostname in ('localhost','127.0.0.1')):
            raise ValueError('Use the HTTPS tunnel URL, or localhost for local jobs')
        self.url=url.rstrip('/'); self.token=token

    def call(self,route,payload=None,binary=False):
        if payload is not None and route!='/heartbeat' and hasattr(self,'lease'):
            payload=dict(payload,_lease=self.lease)
        data=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request(self.url+route,data=data,
            headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request,timeout=45) as response: body=response.read()
                return body if binary else json.loads(body)
            except urllib.error.HTTPError as exc:
                reason=exc.read().decode(errors='replace')[:500]
                if exc.code not in (408,429,500,502,503,504,520,522,524):
                    raise OSError(f'Receiver {exc.code}: {reason}') from None
                if attempt==3: raise OSError(f'Receiver unavailable: {exc.code}') from None
            except (OSError,TimeoutError):
                if attempt==3: raise
            time.sleep(2**attempt)

    def put(self,root,path):
        root,path=Path(root),Path(path); relative=path.relative_to(root).as_posix()
        with path.open('rb') as f:
            import hashlib
            h=hashlib.sha256()
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            sha=h.hexdigest(); f.seek(0)
            init=self.call('/upload/start',{'path':relative,'sha256':sha,'size':path.stat().st_size})
            if init.get('complete'): return
            offset=init['offset']; f.seek(offset)
            while block:=f.read(4*1024*1024):
                self.call('/upload/chunk',{'id':init['id'],'offset':offset,
                    'data':base64.b64encode(block).decode()}); offset+=len(block)
        self.call('/upload/finish',{'id':init['id']})

    def fetch(self,relative,destination):
        from urllib.parse import quote
        info=self.call('/artifact/info?path='+quote(relative,safe=''))
        if not info['exists']: return False
        path=Path(destination); part=path.with_suffix(path.suffix+'.partial')
        path.parent.mkdir(parents=True,exist_ok=True)
        # Restart a download cleanly; server-side acknowledged uploads remain untouched.
        with part.open('wb') as f:
            for offset in range(0,info['size'],4*1024*1024):
                f.write(self.call('/artifact/chunk?path='+quote(relative,safe='')+'&offset='+str(offset),binary=True))
            f.flush(); os.fsync(f.fileno())
        if file_hash(part)!=info['sha256']: raise ValueError('Download hash mismatch')
        os.replace(part,path); return True

def file_hash(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_extract(archive,destination):
    destination=Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination) or (item.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive entry')
        z.extractall(destination)

def package(folder,path):
    folder,path=Path(folder),Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not any(x.startswith('.') for x in p.relative_to(folder).parts):
                z.write(p,p.relative_to(folder).as_posix())
    os.replace(temp,path)

def make_bundle(root):
    root=Path(root); path=root/'runtime/bootstrap.zip'; path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('streamlined','configs','manifests','data','models/tokenizers'):
            for p in (root/directory).rglob('*'):
                if p.is_file() and not any(x in ('source','__pycache__','.cache') for x in p.relative_to(root).parts):
                    z.write(p,p.relative_to(root).as_posix())
        z.write(root/'requirements-colab.txt','requirements-colab.txt')
    return path
