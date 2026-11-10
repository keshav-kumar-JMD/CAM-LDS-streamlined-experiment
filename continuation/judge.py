"""Existing uniform Haiku rubric over one quantized candidate per API context.

The shared rubric is cached on the wire only; its wording and the per-row case
are byte-for-byte the same as the noncached judge construction.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.metadata
import sys
import time
from filelock import FileLock
from streamlined.api import APIRun, client, judge_jobs
from streamlined.analysis import strict_confidence
from streamlined.common import CONDITIONS, POLICY, digest, read_json, write_csv, write_json, verify_experiment
from streamlined.data import records
from streamlined.monitor import Heartbeat
from streamlined.rubrics import rubric
from .protocol import CONVENTIONAL, NEW_JUDGE_RUN, OPTIONAL_JUDGE_RUN, stream

REQUIRED = [('bitnet2b','native_i2_s'), *((key,'tq1_0') for key in CONVENTIONAL)]
OPTIONAL = [(key,'iq1_s_camlds_calibrated') for key in CONVENTIONAL]

def expected_streams(family='required'):
    pairs=REQUIRED if family=='required' else OPTIONAL if family=='optional' else None
    if pairs is None:raise ValueError(family)
    return [(model,representation,adapter,stream(model,representation,adapter))
            for model,representation in pairs for adapter in (None,*CONDITIONS)]

def source_rows(root,family='required',require_complete=False):
    root=Path(root);verify_experiment(root)
    source={str(row['prompt_id']):row for row in records(root)}
    reference_dir=root/'results/api/haiku/records'
    reference={str(r['prompt_id']):r for p in reference_dir.glob('*.json')
               if (r:=read_json(p)).get('status')=='completed'}
    if len(reference)!=990:raise RuntimeError('Haiku reference must contain 990 completed prompts')
    output=[];coverage=[]
    for model,representation,adapter,name in expected_streams(family):
        directory=root/'results'/name/'records'
        index={str(r['prompt_id']):r for p in directory.glob('*.json') if (r:=read_json(p))}
        ids=[pid for pid,row in source.items()
             if (adapter is None and family=='required') or row['split'] in ('validation','test')]
        states=Counter(index.get(pid,{}).get('status','missing') for pid in ids)
        judge_eligible=states.get('completed',0)
        coverage.append(dict(model=model,representation=representation,
            condition='untreated' if adapter is None else 'sft_'+adapter,
            expected=len(ids),judge_eligible=judge_eligible,
            omitted_no_response=len(ids)-judge_eligible,**states))
        unresolved=[pid for pid in ids if index.get(pid,{}).get('status') not in ('completed','context_unsupported')]
        if unresolved and require_complete:
            raise RuntimeError(f'{name}: {len(unresolved)} rows still need inference; first IDs {unresolved[:8]}')
        for pid in ids:
            candidate=index.get(pid,{})
            if candidate.get('status')!='completed':continue
            row=source[pid];ref=reference[pid]
            if candidate.get('prompt_sha256')!=row['prompt_sha256'] or ref.get('prompt_sha256')!=row['prompt_sha256']:
                raise ValueError(f'Prompt identity mismatch for {name} ID {pid}')
            output.append(dict(row,model=model,
                condition='untreated' if adapter is None else 'sft_'+adapter,
                representation=representation,adapter=adapter or '',
                candidate_status='completed',candidate_record_id=candidate['record_id'],
                candidate_response_raw=candidate['response_raw'],
                candidate_response_sha256=candidate['response_hash'],
                candidate_confidence_label_strict=strict_confidence(candidate['response_raw']),
                haiku_reference_status='completed',
                haiku_reference_record_id=ref['record_id'],
                haiku_reference_response_raw=ref['response_raw'],
                haiku_reference_response_sha256=ref['response_hash'],
                **{'candidate_'+k:v for k,v in candidate.items() if k not in
                   ('prompt_id','prompt_text','prompt_sha256','scenario','variant','step',
                    'split','ground_truth','record_id','response_raw','response_hash','status')}))
    return output,coverage

class FastLocalStore:
    """Immediate atomic request/receipt rows; materialize large CSV periodically."""
    def __init__(self,folder):
        self.folder=Path(folder);self.rowdir=self.folder/'records';self.rowdir.mkdir(parents=True,exist_ok=True)
        self.rows={r['record_id']:r for p in self.rowdir.glob('*.json') if (r:=read_json(p))}
        self.dirty=0
    def done(self,record_id):return self.rows.get(record_id,{}).get('status')=='completed'
    def put(self,row):
        with FileLock(str(self.folder/'records.lock')):
            old=self.rows.get(row['record_id'])
            if old and old.get('status')=='completed':
                if row.get('status')!='completed' or old.get('response_hash')!=row.get('response_hash'):
                    raise ValueError('Completed judgment cannot be changed')
                return
            if old and old.get('attempt_count')!=row.get('attempt_count'):
                write_json(self.folder/'attempt_history'/f'{digest(row["record_id"])}_{old.get("attempt_count",0)}.json',old)
            write_json(self.rowdir/(digest(row['record_id'])+'.json'),row)
            self.rows[row['record_id']]=row;self.dirty+=1
            if self.dirty>=100:self.export()
    def export(self):
        write_csv(self.folder/'responses.csv',self.rows.values());self.dirty=0

class CachedJudgeRun(APIRun):
    def __init__(self,root,family='required'):
        # Available-case semantic judging: only completed records contain a
        # returned candidate response. Other statuses remain in coverage and
        # are never converted into synthetic answers or zero judge scores.
        rows,coverage=source_rows(root,family,require_complete=False)
        jobs=judge_jobs(rows)
        if not jobs:raise RuntimeError('No completed quantized candidates to judge')
        unchanged=rubric()
        for job in jobs:
            if job['params']['system']!=unchanged:raise ValueError('Judge rubric differs from frozen common rubric')
            job['params']['system']=[{'type':'text','text':unchanged,
                                     'cache_control':{'type':'ephemeral'}}]
        name=NEW_JUDGE_RUN if family=='required' else OPTIONAL_JUDGE_RUN
        self.root=Path(root);self.name=name;self.api=client(root)
        self.folder=self.root/'results/api'/name;self.folder.mkdir(parents=True,exist_ok=True)
        self.ledger=self.folder/'submissions';self.ledger.mkdir(exist_ok=True)
        manifest=self.folder/'request_manifest.json';marker=self.folder/'request_identity.json'
        if manifest.exists():
            frozen=read_json(manifest);self.jobs=frozen['jobs']
            request_hash=digest(sorted(self.jobs))
            if frozen['hash']!=request_hash:raise ValueError('Frozen judge request manifest is internally inconsistent')
            if marker.exists() and read_json(marker)['hash']!=request_hash:
                raise ValueError('Judge request identity marker differs from its frozen manifest')
        else:
            self.jobs={}
            for job in jobs:
                rid=digest([POLICY['experiment_id'],job['identity'],job['params']])
                self.jobs[rid]=dict(job,record_id=rid)
            request_hash=digest(sorted(self.jobs))
            write_json(manifest,{'hash':request_hash,'jobs':self.jobs,
                'price_snapshot_date':'2026-10-24',
                'price_source':'https://platform.claude.com/docs/en/models/haiku-5-5/overview',
                'temperature':'omitted: provider default; not greedy','seed':None,
                'sdk_version':importlib.metadata.version('anthropic'),'python':sys.version})
        if not marker.exists():write_json(marker,{'hash':request_hash})
        self.store=FastLocalStore(self.folder)
        coverage_path=self.folder/'source_coverage.csv'
        if not coverage_path.exists():write_csv(coverage_path,coverage)
        write_json(self.folder/'cache_policy.json',{
            'family':family,'rubric_sha256':digest(unchanged),
            'cache_block':'static system rubric only','case_policy':'one user case per request',
            'cache_duration':'provider default ephemeral','judge_protocol':'uniform_common_v2_confidence_agreement',
            'candidate_count':len(self.jobs),'request_count':len(self.jobs),
            'note':'Cache hits are opportunistic, especially in batches; usage records retain actual cache token counts.'})
    def poll(self,max_seconds=3600):
        """Rate-paced, idempotent collection for the large notebook 07 ledger."""
        with FileLock(str(self.folder/'run.lock'),timeout=0),Heartbeat(
                self.root,'api/'+self.name+'/poll',stall_seconds=90000) as hb:
            deadline=time.monotonic()+max_seconds;progress={};batch_poll_interval=2.6
            def collect_path(path):
                while True:
                    try:
                        intent=read_json(path);batch=self.api.messages.batches.retrieve(intent['batch_id'])
                        observed=(batch.processing_status,str(batch.request_counts))
                        if progress.get(batch.id)!=observed:
                            progress[batch.id]=observed;hb.update(batch.id,'polling '+batch.processing_status)
                        if batch.processing_status!='ended':return
                        seen=set()
                        for result in self.api.messages.batches.results(batch.id):
                            if result.custom_id not in intent['custom_ids']:raise ValueError('Unexpected remote request identity')
                            rid=intent['custom_ids'][result.custom_id];seen.add(rid);job=self.jobs[rid]
                            receipt=self.folder/'batch_receipts'/batch.id/(rid+'.json');write_json(receipt,result.model_dump())
                            if not self.store.done(rid):
                                if result.result.type=='succeeded':
                                    self.save_message(job,result.result.message,batch=True,seconds=time.time()-intent['created_at'])
                                else:self.status(job,'retryable_error',error=result.result.model_dump())
                        if seen==set(intent['jobs']):intent['state']='collected';write_json(path,intent)
                        return
                    except Exception as error:
                        if getattr(error,'status_code',None)!=429:raise
                        wait_seconds=65;response=getattr(error,'response',None)
                        if response is not None:
                            try:wait_seconds=max(wait_seconds,float(response.headers.get('retry-after',0)))
                            except (TypeError,ValueError):pass
                        hb.update('rate_limit','read-only polling paused for '+str(wait_seconds)+' seconds')
                        print(f'Message Batches API rate limit reached; waiting {wait_seconds:.0f}s and resuming safely.',flush=True)
                        time.sleep(wait_seconds)
            while True:
                pending=[p for p in self.ledger.glob('*.json') if read_json(p)['state']=='pending']
                if not pending:break
                for path in pending:
                    collect_path(path);time.sleep(batch_poll_interval)
                print(self.counts(),flush=True)
                if time.monotonic()>=deadline:
                    if any(read_json(p)['state']=='pending' for p in self.ledger.glob('*.json')):hb.final_state='attention'
                    break
                if not any(read_json(p)['state']=='pending' for p in self.ledger.glob('*.json')):break
                time.sleep(30)
        return self.counts()
    def export(self):
        self.store.export();return self.counts()

def make_run(root,family='required'):
    return CachedJudgeRun(root,family)
