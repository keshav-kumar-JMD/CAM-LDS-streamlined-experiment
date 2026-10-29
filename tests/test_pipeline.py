import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from streamlined.common import *
from streamlined.data import object_spans
from streamlined.encoding import render,encode_exact
from streamlined.monitor import reasons
from streamlined.receiver import handler,resolve
from streamlined.transport import Mirror,safe_extract
from streamlined.storage import Store
from streamlined.api import APIRun,parse_judgment,judge_jobs,COMMON_JUDGE_RUN,HAIKU_SELF_JUDGE_RUN
from streamlined.rubrics import COMMON,rubric

class CharacterTokenizer:
    eos_token='~'
    def apply_chat_template(self,messages,tokenize=False):
        return 'USER:'+messages[0]['content']+'|ASSISTANT:'+messages[1]['content']+'~'
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}

class PipelineTests(unittest.TestCase):
    def test_base_models_bypass_checkpoint_tunnel(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'streamlined/gpu.py').read_text(encoding='utf-8')
        loader=source[source.index('def load_base'):source.index('def run_identity')]
        self.assertIn("cache_source='colab_local_reuse' if intact else 'huggingface_direct'",loader)
        self.assertIn('snapshot_download(',loader)
        self.assertNotIn('mirror.fetch(',loader)
        self.assertNotIn('persist(',loader)
        self.assertNotIn('package(',loader)

    def test_exact_source_and_masks(self):
        line=' {"response": { "explanation" : "a\\nb", "techniques": ["T1000"], "confidence":"Neutral" }, "x":2}'
        a,b=object_spans(line)['response'];answer=line[a:b]
        self.assertTrue(answer.startswith('{ "explanation" :'))
        row={'prompt_id':'1','prompt_text':'original\nlog','teacher_response_raw':answer}
        tok=CharacterTokenizer();prefix,full,ending=render(tok,row['prompt_text'],answer)
        self.assertEqual(full,prefix+answer+ending)
        encoded=encode_exact(tok,row)
        decoded=''.join(chr(t) for t in encoded['input_ids'])
        self.assertIn(answer,decoded)
        for field in (1,2,3,4):self.assertIn(field,encoded['fields'])
        self.assertTrue(all(x==-100 for x in encoded['labels'][:len(prefix)]))

    def test_audit_does_not_reject_bad_content(self):
        self.assertFalse(audit_response('nonsense')['valid_json'])
        self.assertFalse(audit_response('{"a":1,"a":2}')['valid_json'])
        self.assertTrue(audit_response('```json\n{"a":1}\n```')['unexpected_text_present'])
        self.assertFalse(parse_judgment('nonsense')['judge_parseable'])
        self.assertEqual(parse_judgment('text {"scores":{"a":3}}')['score_a'],3)

    def test_uniform_rubric_confidence_and_malformed_candidate(self):
        self.assertIn('confidence_evidence_support',COMMON)
        self.assertNotIn('uncertainty_handling',COMMON)
        self.assertIn('confidence_label_agreement',rubric())
        row={'prompt_id':'1','prompt_text':'prompt','prompt_sha256':'hash','ground_truth':['T1000'],
             'model':'qwen','condition':'full','candidate_status':'completed',
             'candidate_record_id':'candidate-one','candidate_response_raw':'not valid JSON',
             'teacher_response_raw':'{}','teacher_confidence_label':'Neutral'}
        jobs=judge_jobs([row]);self.assertEqual(len(jobs),1)
        case=json.loads(jobs[0]['params']['messages'][0]['content'])
        self.assertFalse(case['candidate_format_audit']['valid_json'])
        self.assertEqual(case['candidate_response'],'not valid JSON')
        self.assertNotIn('teacher_response_raw',case)
        parsed=parse_judgment(json.dumps({'scores':{'confidence_evidence_support':3},
            'candidate_confidence_label':'Somewhat Certain: Attack',
            'judge_warranted_confidence_label':'Almost Certain: Attack',
            'confidence_label_agreement':3,'confidence_label_agreement_reason':'adjacent',
            'confidence_label_null_reason':None}))
        self.assertEqual(parsed['candidate_confidence_label'],'Somewhat Certain: Attack')
        self.assertEqual(parsed['confidence_label_agreement'],3)

    def test_csv_multiline_and_completed_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            store=Store(td,'test');row={'record_id':'one','status':'completed','response_raw':'"quoted"\nline',
                                       'response_hash':digest('"quoted"\nline')}
            store.put(row);store=Store(td,'test')
            self.assertTrue(store.done('one'));self.assertEqual(read_csv(store.folder/'responses.csv')[0]['response_raw'],row['response_raw'])
            with self.assertRaises(ValueError):store.put(dict(row,response_hash='different'))
            with self.assertRaises(ValueError):store.put(dict(row,status='unknown'))

    def test_local_unsent_response_survives_stale_remote_state(self):
        class Stale:
            def call(self,*args,**kwargs):
                return {'rows':[{'record_id':'one','status':'in_progress','response_hash':''}]}
        with tempfile.TemporaryDirectory() as td:
            store=Store(td,'test');store.put({'record_id':'one','status':'completed',
                'response_raw':'saved locally','response_hash':digest('saved locally')})
            recovered=Store(td,'test',Stale())
            self.assertTrue(recovered.done('one'))
            self.assertEqual(recovered.rows['one']['response_raw'],'saved locally')

    def test_watchdog_stale_and_stalled(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'runtime/heartbeats/a.json'
            row={'run_id':'run','owner':'one','state':'running','received_at':100,'last_progress':100,'stall_seconds':500}
            write_json(p,row);self.assertFalse(reasons(td,200));self.assertTrue(reasons(td,251))
            write_json(p,dict(row,received_at=900));self.assertIn('stalled',reasons(td,900)[0][1])
            write_json(p,dict(row,state='completed'));self.assertFalse(reasons(td,1000))

    def test_receiver_roundtrip_retry_and_lease(self):
        with tempfile.TemporaryDirectory() as td,tempfile.TemporaryDirectory() as remote:
            root=Path(td);(root/'runtime').mkdir();server=ThreadingHTTPServer(('127.0.0.1',0),handler(root,'test-token'))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                mirror=Mirror(f'http://127.0.0.1:{server.server_port}','test-token')
                source=Path(remote)/'models/test.bin';source.parent.mkdir();source.write_bytes(b'abc'*100000)
                mirror.put(remote,source);mirror.put(remote,source)
                self.assertEqual((root/'models/test.bin').read_bytes(),source.read_bytes())
                dest=Path(remote)/'restored.bin';self.assertTrue(mirror.fetch('models/test.bin',dest))
                self.assertEqual(dest.read_bytes(),source.read_bytes())
                store=Store(remote,'qwen/full/test',mirror)
                row={'record_id':'r','status':'completed','response_raw':'bad JSON','response_hash':digest('bad JSON')}
                store.put(row)
                # Destroy the materialized CSV: record log must restore it from authoritative JSON.
                (root/'results/qwen/full/test/responses.csv').unlink()
                restored=Store(remote,'qwen/full/test',mirror);self.assertTrue(restored.done('r'))
                mirror.call('/heartbeat',{'run_id':'r','owner':'a','state':'running'})
                with self.assertRaises(OSError):mirror.call('/heartbeat',{'run_id':'r','owner':'b','state':'running'})
                with self.assertRaises(ValueError):resolve(root,'../secret/key')
            finally:server.shutdown();server.server_close();thread.join()

    def test_api_receipt_retains_invalid_output_and_nulls(self):
        with tempfile.TemporaryDirectory() as td:
            job={'identity':'one','metadata':{'prompt_id':'1','model':'qwen','condition':'full',
                 'candidate_valid_json':True,'teacher_confidence_label':'Neutral',
                 'haiku_reference_response_raw':'reference'},
                 'params':{'model':'fake','max_tokens':10,'messages':[]}}
            run=APIRun(td,'common',[job],api=object());item=next(iter(run.jobs.values()))
            run.save_message(item,{'content':[{'type':'text','text':'not json'}], 'usage':{},'stop_reason':'end_turn'})
            self.assertEqual(run.counts()['completed'],1);self.assertFalse(run.eligible(True))
            self.assertEqual(run.store.rows[item['record_id']]['response_raw'],'not json')
            self.assertFalse(run.store.rows[item['record_id']]['judge_parseable'])
            self.assertFalse(run.store.rows[item['record_id']]['judge_valid_json'])
            self.assertTrue(run.store.rows[item['record_id']]['candidate_valid_json'])
            self.assertEqual(run.store.rows[item['record_id']]['teacher_confidence_label'],'Neutral')

    def test_changed_paid_request_manifest_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            one={'identity':'one','metadata':{'prompt_id':'1'},'params':{'model':'x','max_tokens':10}}
            APIRun(td,COMMON_JUDGE_RUN,[one],api=object())
            changed={'identity':'one','metadata':{'prompt_id':'1'},'params':{'model':'x','max_tokens':11}}
            with self.assertRaises(ValueError):APIRun(td,COMMON_JUDGE_RUN,[changed],api=object())

    def test_ambiguous_paid_submission_is_held(self):
        class FakeBatches:
            def create(self,**kwargs):raise TimeoutError('connection lost')
        with tempfile.TemporaryDirectory() as td:
            job={'identity':'one','metadata':{'prompt_id':'1'},'params':{'max_tokens':10}}
            api=SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
            run=APIRun(td,'haiku',[job],api=api)
            with self.assertRaises(TimeoutError):run.submit()
            self.assertEqual(run.counts()['unknown'],1)
            self.assertFalse(run.eligible(True))
            with self.assertRaises(RuntimeError):run.submit()

    def test_batch_retry_custom_ids_are_distinct(self):
        class Rejected(Exception):status_code=429
        class FakeBatches:
            ids=[]
            def create(self,requests):
                self.ids.append(requests[0]['custom_id']);raise Rejected('rate limited')
        with tempfile.TemporaryDirectory() as td:
            batches=FakeBatches();api=SimpleNamespace(messages=SimpleNamespace(batches=batches))
            job={'identity':'one','metadata':{'prompt_id':'1'},'params':{'max_tokens':10}}
            run=APIRun(td,'haiku',[job],api=api)
            with self.assertRaises(Rejected):run.submit()
            with self.assertRaises(Rejected):run.submit(retry_failed=True)
            self.assertNotEqual(batches.ids[0],batches.ids[1])

    def test_notebooks_valid_and_recovery_documented(self):
        import nbformat
        folder=Path(__file__).resolve().parents[1]/'notebooks'
        files=list(folder.glob('*.ipynb'));self.assertEqual(len(files),14)
        for path in files:
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
            full_text='\n'.join(c.source for c in nb.cells)
            if path.name=='00_prepare_universal_dataset.ipynb':
                self.assertIn('cloudflared tunnel',full_text)
                self.assertIn('streamlined.receiver --root',full_text)
            else:
                self.assertNotIn('cloudflared tunnel',full_text)
                self.assertNotIn('streamlined.receiver --root',full_text)
            for i,c in enumerate(nb.cells):
                if c.cell_type=='code':
                    compile(c.source,str(path),'exec')
                    if c.get('id','').startswith('code-'):
                        description=next(nb.cells[j].source for j in range(i-1,-1,-1) if nb.cells[j].cell_type=='markdown')
                        self.assertIn('Recovery / safe rerun',description)

if __name__=='__main__':unittest.main()
