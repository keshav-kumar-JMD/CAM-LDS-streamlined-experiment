"""Existing uniform Haiku rubric over one quantized candidate per API context.

The shared rubric is cached on the wire only; its wording and the per-row case
are byte-for-byte the same as the noncached judge construction.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.metadata
import sys
from filelock import FileLock
from streamlined.api import APIRun, client, judge_jobs
from streamlined.analysis import strict_confidence
from streamlined.common import CONDITIONS, POLICY, digest, read_json, write_csv, write_json, verify_experiment
from streamlined.data import records
from streamlined.rubrics import rubric
from .protocol import CONVENTIONAL, NEW_JUDGE_RUN, OPTIONAL_JUDGE_RUN, stream

REQUIRED = [('bitnet2b','native_i2_s'), *((key,'tq1_0') for key in CONVENTIONAL)]
OPTIONAL = [(key,'iq1_s_camlds_calibrated') for key in CONVENTIONAL]

def expected_streams(family='required'):
    pairs=REQUIRED if family=='required' else OPTIONAL if family=='optional' else None
    if pairs is None:raise ValueError(family)
    return [(model,representation,adapter,stream(model,representation,adapter))
            for model,representation in pairs for adapter in (None,*CONDITIONS)]

def source_rows(root,family='required',require_complete=True):
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
        coverage.append(dict(model=model,representation=representation,
            condition='untreated' if adapter is None else 'sft_'+adapter,
            expected=len(ids),**states))
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
        rows,coverage=source_rows(root,family)
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
        self.jobs={}
        for job in jobs:
            rid=digest([POLICY['experiment_id'],job['identity'],job['params']])
            self.jobs[rid]=dict(job,record_id=rid)
        self.ledger=self.folder/'submissions';self.ledger.mkdir(exist_ok=True)
        manifest=self.folder/'request_manifest.json';marker=self.folder/'request_identity.json'
        request_hash=digest(sorted(self.jobs))
        prior=read_json(marker)['hash'] if marker.exists() else read_json(manifest)['hash'] if manifest.exists() else None
        if prior and prior!=request_hash:
            raise ValueError('Judge request set changed; the existing paid ledger cannot be reused')
        if not manifest.exists():
            write_json(manifest,{'hash':request_hash,'jobs':self.jobs,
                'price_snapshot_date':'2026-10-24',
                'price_source':'https://platform.claude.com/docs/en/models/haiku-5-5/overview',
                'temperature':'omitted: provider default; not greedy','seed':None,
                'sdk_version':importlib.metadata.version('anthropic'),'python':sys.version})
        if not marker.exists():write_json(marker,{'hash':request_hash})
        self.store=FastLocalStore(self.folder)
        write_csv(self.folder/'source_coverage.csv',coverage)
        write_json(self.folder/'cache_policy.json',{
            'family':family,'rubric_sha256':digest(unchanged),
            'cache_block':'static system rubric only','case_policy':'one user case per request',
            'cache_duration':'provider default ephemeral','judge_protocol':'uniform_common_v2_confidence_agreement',
            'candidate_count':len(rows),'request_count':len(jobs),
            'note':'Cache hits are opportunistic, especially in batches; usage records retain actual cache token counts.'})
    def export(self):
        self.store.export();return self.counts()

def make_run(root,family='required'):
    return CachedJudgeRun(root,family)
