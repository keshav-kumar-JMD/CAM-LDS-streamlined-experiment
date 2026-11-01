"""Recovery invariants for the new local checkpoint protocol."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from continuation.store import SyncStore, accept, record_path
from streamlined.common import read_csv, read_json, write_json

class MirrorStub:
    def __init__(self,root):self.root=Path(root);self.calls=[]
    def call(self,route,payload=None):
        self.calls.append(route)
        stream=(payload or {}).get('stream','quantized/qwen1b/untreated/tq1_0')
        if route.startswith('/continuation/responses?'):
            directory=self.root/'results/quantized/qwen1b/untreated/tq1_0/records'
            return {'rows':[read_json(p) for p in directory.glob('*.json')]}
        if route=='/continuation/response':return {'rows':accept(self.root,stream,[payload['row']])}
        if route=='/continuation/response_batch':return {'rows':accept(self.root,stream,payload['rows'])}
        if route=='/continuation/snapshot':return {'rows':accept(self.root,stream,[],final=True)}
        raise AssertionError(route)

class StoreTests(unittest.TestCase):
    stream='quantized/qwen1b/untreated/tq1_0'
    def row(self,n):
        return {'prompt_id':str(n),'record_id':f'config:{n}','config_hash':'config',
                'status':'completed','response_raw':f'answer {n}','response_hash':f'hash {n}'}
    def test_atomic_rows_survive_between_csv_snapshots(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for n in range(1,10):accept(root,self.stream,[self.row(n)])
            self.assertEqual(len(list((root/'results'/self.stream/'records').glob('*.json'))),9)
            self.assertFalse((root/'results'/self.stream/'responses.csv').exists())
            accept(root,self.stream,[self.row(10)])
            self.assertEqual(len(read_csv(root/'results'/self.stream/'responses.csv')),10)
            accept(root,self.stream,[self.row(11)])
            self.assertEqual(len(read_csv(root/'results'/self.stream/'responses.csv')),10)
            retry=dict(self.row(12),status='retryable_error',response_hash='')
            accept(root,self.stream,[retry])
            accept(root,self.stream,[self.row(12)])
            for n in range(13,20):accept(root,self.stream,[self.row(n)])
            self.assertEqual(len(read_csv(root/'results'/self.stream/'responses.csv')),19)
            accept(root,self.stream,[self.row(20)])
            self.assertEqual(len(read_csv(root/'results'/self.stream/'responses.csv')),19)
            accept(root,self.stream,[],final=True)
            self.assertEqual(len(read_csv(root/'results'/self.stream/'responses.csv')),20)
            changed=dict(self.row(11),response_hash='different')
            with self.assertRaises(ValueError):accept(root,self.stream,[changed])
    def test_local_spool_is_replayed_before_resume(self):
        with TemporaryDirectory() as remote,TemporaryDirectory() as local:
            row=self.row(3)
            target=record_path(local,self.stream,row['record_id'])
            write_json(target,row)
            mirror=MirrorStub(remote)
            store=SyncStore(local,self.stream,mirror)
            self.assertTrue(store.done(row['record_id']))
            self.assertTrue(record_path(remote,self.stream,row['record_id']).is_file())
            self.assertIn('/continuation/response_batch',mirror.calls)
            result=store.put(row)
            self.assertEqual(result['receiver_seconds'],0)

if __name__=='__main__':unittest.main()
