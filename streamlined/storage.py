from pathlib import Path
from urllib.parse import quote
from filelock import FileLock
from .common import *

class Store:
    def __init__(self,root,stream,mirror=None):
        self.root=Path(root);self.stream=stream;self.mirror=mirror
        self.folder=self.root/'results'/stream;self.folder.mkdir(parents=True,exist_ok=True)
        self.rowdir=self.folder/'records';self.rowdir.mkdir(exist_ok=True)
        if mirror:
            for row in mirror.call('/responses?stream='+quote(stream,safe=''))['rows']:
                path=self.rowdir/(digest(row['record_id'])+'.json')
                local=read_json(path) if path.exists() else {}
                if local.get('status')=='completed':
                    if row.get('status')=='completed' and local.get('response_hash')!=row.get('response_hash'):
                        raise ValueError('Local and receiver completed responses conflict')
                    continue
                write_json(path,row)
        self.rows={r['record_id']:r for p in self.rowdir.glob('*.json') if (r:=read_json(p))}
        self.export()
    def export(self):write_csv(self.folder/'responses.csv',self.rows.values())
    def put(self,row):
        with FileLock(str(self.folder/'records.lock')):
            prev=self.rows.get(row['record_id'])
            if prev and prev.get('status')=='completed':
                if row.get('status')!='completed' or prev.get('response_hash')!=row.get('response_hash'):
                    raise ValueError('Refusing to change or downgrade a completed record')
            if prev and prev.get('attempt_count')!=row.get('attempt_count'):
                write_json(self.folder/'attempt_history'/f'{digest(row["record_id"])}_{prev.get("attempt_count",0)}.json',prev)
            write_json(self.rowdir/(digest(row['record_id'])+'.json'),row)
            self.rows[row['record_id']]=row;self.export()
            if self.mirror:self.mirror.call('/response',{'stream':self.stream,'row':row})
    def done(self,record_id):
        row=self.rows.get(record_id,{})
        return row.get('status') in ('completed','context_unsupported')

def persist(root,path,mirror=None):
    if mirror:mirror.put(root,path)
