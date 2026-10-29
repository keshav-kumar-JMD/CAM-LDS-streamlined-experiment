"""Local paid API ledger. Returned content is always retained, never quality-retried."""
from __future__ import annotations
import json
from pathlib import Path
import time
import uuid
import sys
import importlib.metadata
from filelock import FileLock
from .common import *
from .data import records
from .monitor import Heartbeat
from .storage import Store
from .telemetry import cost
from .rubrics import rubric

COMMON_JUDGE_RUN = 'common_uniform_v2'
HAIKU_SELF_JUDGE_RUN = 'haiku_self_uniform_v2'
JUDGE_PROTOCOL = 'uniform_common_v2_confidence_agreement'

def client(root):
    import anthropic
    return anthropic.Anthropic(api_key=secret(root,'Claude_API_Key.txt'),max_retries=0,timeout=900)

def haiku_jobs(root):
    verify_experiment(root)
    jobs=[]
    for row in records(root):
        jobs.append({'identity':f'haiku:{row["prompt_id"]}','metadata':dict(row,model='haiku',condition='haiku'),
                     'params':{'model':POLICY['haiku_model'],'max_tokens':POLICY['haiku_max_tokens'],
                               'thinking':POLICY['haiku_thinking'],
                               'messages':[{'role':'user','content':row['prompt_text']}]}})
    return jobs

def judge_jobs(rows,self_assessment=False):
    jobs=[]
    for row in rows:
        if row.get('candidate_status',row.get('status'))!='completed':continue
        # Audit quality never controls inclusion: every completed candidate is judged verbatim.
        candidate=row.get('candidate_response_raw',row.get('response_raw',''))
        candidate_audit=audit_response(candidate)
        case={'prompt':row['prompt_text'],'ground_truth':row['ground_truth'],
              'candidate_response':candidate,'candidate_format_audit':candidate_audit}
        if not self_assessment:
            case['haiku_reference_response']=row.get('haiku_reference_response_raw',row.get('haiku_response',''))
        candidate_record_id=row.get('candidate_record_id',row.get('record_id',''))
        metadata=dict(row,judge_protocol=JUDGE_PROTOCOL,
            candidate_status=row.get('candidate_status',row.get('status','')),
            candidate_record_id=candidate_record_id,candidate_response_raw=candidate,
            candidate_response_sha256=digest(candidate),
            **{'candidate_'+key:value for key,value in candidate_audit.items()})
        jobs.append({'identity':f'judge:{JUDGE_PROTOCOL}:{candidate_record_id}',
            'metadata':metadata,
            'params':{'model':POLICY['haiku_model'],'max_tokens':POLICY['judge_max_tokens'],
                'thinking':POLICY['haiku_thinking'],'system':rubric(),
                'messages':[{'role':'user','content':json.dumps(case,ensure_ascii=False)}]}})
    return jobs

def parse_judgment(text):
    """Extract available fields only; preserve raw output regardless of parser result."""
    obj=None
    try:obj=strict_json(text)
    except (ValueError,TypeError):
        decoder=json.JSONDecoder()
        for i,c in enumerate(text):
            if c=='{':
                try:
                    value,_=decoder.raw_decode(text,i)
                    if isinstance(value,dict):obj=value;break
                except ValueError:pass
    result={'judge_parseable':isinstance(obj,dict),'judge_parsed':obj}
    direct={'candidate_confidence_label','judge_warranted_confidence_label',
            'confidence_label_agreement','confidence_label_agreement_reason','confidence_label_null_reason'}
    if isinstance(obj,dict):
        for name,value in obj.items():
            if name=='scores' and isinstance(value,dict):
                for metric,score in value.items():result['score_'+metric]=score
            elif name in direct:result[name]=value
            else:result['judge_'+name]=value
    return result

class APIRun:
    def __init__(self,root,name,jobs,api=None):
        self.root=Path(root);self.name=name;self.api=api or client(root)
        self.folder=self.root/'results/api'/name;self.folder.mkdir(parents=True,exist_ok=True)
        self.jobs={}
        for job in jobs:
            rid=digest([POLICY['experiment_id'],job['identity'],job['params']])
            self.jobs[rid]=dict(job,record_id=rid)
        self.store=Store(root,'api/'+name);self.ledger=self.folder/'submissions';self.ledger.mkdir(exist_ok=True)
        # Prevent accidental reuse of one ledger with changed prompts/rubric/model.
        frozen=self.folder/'request_manifest.json';identity=digest(sorted(self.jobs))
        if frozen.exists() and read_json(frozen)['hash']!=identity:
            raise ValueError('API request set changed. Finish compilation before creating this run, or use a new run name.')
        write_json(frozen,{'hash':identity,'jobs':self.jobs,'price_snapshot_date':'2026-10-24',
            'price_source':'https://platform.claude.com/docs/en/models/haiku-5-5/overview',
            'temperature':'omitted: provider default; not greedy','seed':None,
            'sdk_version':importlib.metadata.version('anthropic'),'python':sys.version})
    def counts(self):
        from collections import Counter
        states=Counter(self.store.rows.get(k,{}).get('status','pending') for k in self.jobs)
        return dict(total=len(self.jobs),**states)
    def eligible(self,retry_failed=False):
        allowed={'pending'}|({'rejected','retryable_error'} if retry_failed else set())
        return [j for rid,j in self.jobs.items() if self.store.rows.get(rid,{}).get('status','pending') in allowed]
    def status(self,job,state,**extras):
        previous=self.store.rows.get(job['record_id'],{})
        self.store.put(dict(previous,record_id=job['record_id'],status=state,
            prompt_id=job['metadata']['prompt_id'],model=job['metadata'].get('model',''),
            condition=job['metadata'].get('condition',''),**extras))
    def save_message(self,job,message,batch=False,seconds=None):
        # message.model_dump includes usage, request ID when provided, all blocks and stop reason.
        response=message.model_dump() if hasattr(message,'model_dump') else message
        text=''.join(b.get('text','') for b in response.get('content',[]) if b.get('type')=='text')
        usage=response.get('usage',{});previous=self.store.rows.get(job['record_id'],{})
        audit=audit_response(text)
        out=dict(job['metadata'],record_id=job['record_id'],status='completed',response_raw=text,
            response_hash=digest(text),finish_reason=response.get('stop_reason'),
            response_truncated=response.get('stop_reason')=='max_tokens',api_response=response,
            usage=usage,estimated_cost_usd=cost(usage,batch),batch_api=batch,
            elapsed_seconds=seconds,latency_note='batch completion delay; inference latency unavailable' if batch else 'request wall time',
            attempt_count=previous.get('attempt_count',1),input_tokens=usage.get('input_tokens'),
            output_tokens=usage.get('output_tokens'),wire_max_tokens=job['params']['max_tokens'],
            api_request_id=getattr(message,'_request_id',None))
        if self.name=='haiku':
            out.update(audit)
            out.update(haiku_reference_response_raw=text,
                       haiku_reference_finish_reason=response.get('stop_reason'),
                       **{'haiku_reference_'+key:value for key,value in audit.items()})
        else:
            out.update(judge_response_raw=text,judge_response_sha256=digest(text),
                       judge_finish_reason=response.get('stop_reason'),
                       judge_response_truncated=response.get('stop_reason')=='max_tokens',
                       **{'judge_'+key:value for key,value in audit.items()})
            out.update(parse_judgment(text))
        self.store.put(out)
    def pilot(self,size=5,retry_failed=False):
        with FileLock(str(self.folder/'run.lock'),timeout=0),Heartbeat(self.root,'api/'+self.name+'/pilot',stall_seconds=1200) as hb:
            chosen=self.eligible(retry_failed)[:size];print('Paid pilot requests:',len(chosen),flush=True)
            for job in chosen:
                previous=self.store.rows.get(job['record_id'],{})
                self.status(job,'unknown',attempt_count=int(previous.get('attempt_count',0))+1)
                start=time.perf_counter();hb.update(job['record_id'],'API request')
                try:
                    with self.api.messages.stream(**job['params']) as stream:message=stream.get_final_message()
                    receipt=self.folder/'receipts'/f'{job["record_id"]}.json'
                    write_json(receipt,message.model_dump());self.save_message(job,message,seconds=time.perf_counter()-start)
                    print(job['metadata']['prompt_id'],self.store.rows[job['record_id']]['response_raw'],flush=True)
                    print('Usage/cost:',self.store.rows[job['record_id']]['usage'],self.store.rows[job['record_id']]['estimated_cost_usd'])
                    hb.update(job['record_id'],'saved locally')
                except BaseException as error:
                    code=getattr(error,'status_code',None)
                    self.status(job,'rejected' if code in (400,401,403,404,413,422,429) else 'unknown',
                                error=f'{type(error).__name__}: {error}'[:1000])
                    raise
        return self.counts()
    def restore_receipts(self):
        for path in (self.folder/'receipts').glob('*.json'):
            if path.stem in self.jobs and not self.store.done(path.stem):
                self.save_message(self.jobs[path.stem],read_json(path))
    def resolve_unknown_pilot(self,record_id,reason,receipt_path=None,allow_retry=False):
        """Explicit audited operator action; a retry may incur a duplicate charge."""
        if not reason.strip():raise ValueError('Document the investigation/resolution reason')
        with FileLock(str(self.folder/'run.lock'),timeout=0):
            row=self.store.rows.get(record_id,{})
            if row.get('status')!='unknown':raise ValueError('This record is not an unknown request')
            if any(record_id in read_json(p)['jobs'] for p in self.ledger.glob('*.json')):
                raise ValueError('This is a batch request; use batch reconciliation, not pilot resolution')
            write_json(self.folder/'resolutions'/f'{record_id}_{time.time_ns()}.json',
                {'record_id':record_id,'reason':reason,'receipt_path':str(receipt_path) if receipt_path else None,
                 'allow_retry':allow_retry,'warning':'Retrying an unknown synchronous outcome may duplicate charges'})
            if receipt_path:self.save_message(self.jobs[record_id],read_json(receipt_path))
            elif allow_retry:self.status(self.jobs[record_id],'retryable_error',resolution_reason=reason)
            else:raise ValueError('Supply a recovered API receipt or explicitly allow a retry')
    def submit(self,batch_size=50,retry_failed=False):
        with FileLock(str(self.folder/'run.lock'),timeout=0),Heartbeat(self.root,'api/'+self.name+'/submit',stall_seconds=1200) as hb:
            self.restore_receipts()
            uncertain=[p for p in self.ledger.glob('*.json') if read_json(p)['state']=='unknown']
            if uncertain:raise RuntimeError('Uncertain batch submission: run reconciliation before submitting more')
            jobs=self.eligible(retry_failed);print('Paid batch requests:',len(jobs),flush=True)
            for start in range(0,len(jobs),batch_size):
                chunk=jobs[start:start+batch_size];sid=uuid.uuid4().hex;path=self.ledger/(sid+'.json')
                custom_ids={digest([sid,j['record_id']]):j['record_id'] for j in chunk}
                intent={'state':'unknown','created_at':time.time(),'jobs':[j['record_id'] for j in chunk],
                        'custom_ids':custom_ids}
                write_json(path,intent)
                for job in chunk:
                    prior=self.store.rows.get(job['record_id'],{})
                    self.status(job,'unknown',attempt_count=int(prior.get('attempt_count',0))+1)
                try:
                    batch=self.api.messages.batches.create(requests=[{'custom_id':cid,'params':self.jobs[rid]['params']}
                                                                    for cid,rid in custom_ids.items()])
                    # Save remote ID before any other mutation. Recovery can then poll safely.
                    intent.update(state='pending',batch_id=batch.id);write_json(path,intent)
                    for job in chunk:self.status(job,'submitted',batch_id=batch.id)
                    hb.update(batch.id,'batch submitted');print('Submitted',batch.id,len(chunk),flush=True)
                except BaseException as error:
                    code=getattr(error,'status_code',None)
                    if code in (400,401,403,404,413,422,429):
                        intent['state']='rejected';write_json(path,intent)
                        for job in chunk:self.status(job,'rejected',error=str(error)[:1000])
                    raise
        return self.counts()
    def reconcile(self):
        """Match completed remote batches by exact custom-ID set; never guess from count."""
        with FileLock(str(self.folder/'run.lock'),timeout=0),Heartbeat(self.root,'api/'+self.name+'/reconcile',stall_seconds=1800) as hb:
            self.restore_receipts()
            unknown=[p for p in self.ledger.glob('*.json') if read_json(p)['state']=='unknown']
            if not unknown:return self.counts()
            earliest=min(read_json(p)['created_at'] for p in unknown)
            for batch in self.api.messages.batches.list(limit=100):
                if batch.created_at.timestamp()<earliest-300:break
                hb.update(batch.id,'reconciling')
                if batch.processing_status!='ended':continue
                responses=list(self.api.messages.batches.results(batch.id));ids={r.custom_id for r in responses}
                for path in list(unknown):
                    intent=read_json(path)
                    if ids==set(intent['custom_ids']):
                        intent.update(state='pending',batch_id=batch.id);write_json(path,intent)
                        for rid in intent['jobs']:self.status(self.jobs[rid],'submitted',batch_id=batch.id)
                        unknown.remove(path)
            if unknown:print('Submission outcome still uncertain. Wait for remote completion and reconcile again; no duplicate submission.',flush=True)
        return self.counts()
    def poll(self,max_seconds=3600):
        with FileLock(str(self.folder/'run.lock'),timeout=0),Heartbeat(self.root,'api/'+self.name+'/poll',stall_seconds=90000) as hb:
            deadline=time.monotonic()+max_seconds;progress={}
            while True:
                pending=[p for p in self.ledger.glob('*.json') if read_json(p)['state']=='pending']
                if not pending:break
                for path in pending:
                    intent=read_json(path);batch=self.api.messages.batches.retrieve(intent['batch_id'])
                    observed=(batch.processing_status,str(batch.request_counts))
                    if progress.get(batch.id)!=observed:
                        progress[batch.id]=observed;hb.update(batch.id,'polling '+batch.processing_status)
                    if batch.processing_status!='ended':continue
                    seen=set()
                    for result in self.api.messages.batches.results(batch.id):
                        if result.custom_id not in intent['custom_ids']:raise ValueError('Unexpected remote request identity')
                        rid=intent['custom_ids'][result.custom_id]
                        seen.add(rid);job=self.jobs[rid]
                        receipt=self.folder/'batch_receipts'/batch.id/(rid+'.json');write_json(receipt,result.model_dump())
                        if not self.store.done(rid):
                            if result.result.type=='succeeded':
                                self.save_message(job,result.result.message,batch=True,seconds=time.time()-intent['created_at'])
                            else:self.status(job,'retryable_error',error=result.result.model_dump())
                    if seen==set(intent['jobs']):intent['state']='collected';write_json(path,intent)
                print(self.counts(),flush=True)
                if time.monotonic()>=deadline:
                    if any(read_json(p)['state']=='pending' for p in self.ledger.glob('*.json')):hb.final_state='attention'
                    break
                if not any(read_json(p)['state']=='pending' for p in self.ledger.glob('*.json')):break
                time.sleep(30)
        return self.counts()
