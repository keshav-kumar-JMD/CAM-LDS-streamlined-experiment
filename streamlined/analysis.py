"""Lossless master tables and matched scenario-step cluster intervals."""
from pathlib import Path
import json
import random
from collections import defaultdict
from .common import *
from .data import records
from .storage import Store
from .api import COMMON_JUDGE_RUN, HAIKU_SELF_JUDGE_RUN

def strict_confidence(text):
    """Descriptive strict extraction only; the judge separately recovers malformed content."""
    try:
        value=strict_json(text)
        label=value.get('confidence') if isinstance(value,dict) else None
        return label if label in CONFIDENCES else ''
    except (ValueError,TypeError):return ''

def compile_tables(root):
    root=Path(root);verify_experiment(root);source=records(root);haiku=Store(root,'api/haiku')
    reference={r['prompt_id']:r for r in haiku.rows.values() if r.get('status')=='completed'}
    all_rows=[]
    for model in MODELS:
        for condition in ('untreated_hf',*CONDITIONS):
            parts=('all',) if condition=='untreated_hf' else ('validation','test')
            for part in parts:
                store=Store(root,f'{model}/{condition}/{part}')
                index={r['prompt_id']:r for r in store.rows.values()}
                if len(index)!=len(store.rows):raise ValueError('Multiple protocol versions in candidate stream')
                for row in source:
                    if part!='all' and row['split']!=part:continue
                    actual=index.get(row['prompt_id'],{})
                    if actual and actual['prompt_sha256']!=row['prompt_sha256']:raise ValueError('Candidate prompt mismatch')
                    ref=reference.get(row['prompt_id'],{})
                    if ref and ref['prompt_sha256']!=row['prompt_sha256']:raise ValueError('Haiku prompt mismatch')
                    source_keys=('prompt_id','scenario','variant','step','split','group_id','prompt_text',
                                 'prompt_sha256','ground_truth','teacher_model','teacher_response_raw',
                                 'teacher_response_sha256')
                    out={k:row[k] for k in source_keys}
                    out['teacher_confidence_label']=strict_confidence(row['teacher_response_raw'])
                    out.update(model=model,condition=condition,
                               candidate_status=actual.get('status','pending'),
                               candidate_record_id=actual.get('record_id',''),
                               candidate_response_raw=actual.get('response_raw',''),
                               candidate_response_sha256=actual.get('response_hash',''),
                               candidate_confidence_label_strict=strict_confidence(actual.get('response_raw','')))
                    for key,value in actual.items():
                        if key not in source_keys and key not in ('status','record_id','response_raw','response_hash'):
                            out['candidate_'+key]=value
                    out.update(
                        haiku_reference_status=ref.get('status','pending'),
                        haiku_reference_record_id=ref.get('record_id',''),
                        haiku_reference_response_raw=ref.get('response_raw',''),
                        haiku_reference_response_sha256=ref.get('response_hash',''),
                        haiku_reference_confidence_label_strict=strict_confidence(ref.get('response_raw','')),
                        haiku_reference_finish_reason=ref.get('finish_reason',''))
                    all_rows.append(out)
    output=root/'results/compiled';write_csv(output/'D_all_responses.csv',all_rows)
    for condition in ('technique','explanation','joint'):
        write_csv(output/f'E_{condition}_validation.csv',
                  [r for r in all_rows if r['condition']==condition and r['split']=='validation'])
    coverage=[]
    for model in MODELS:
        for condition in ('untreated_hf',*CONDITIONS):
            for split in ('train','validation','test'):
                rs=[r for r in all_rows if r['model']==model and r['condition']==condition and r['split']==split]
                if rs:coverage.append(dict(model=model,condition=condition,split=split,total=len(rs),
                    completed=sum(r['candidate_status']=='completed' for r in rs),
                    context_unsupported=sum(r['candidate_status']=='context_unsupported' for r in rs),
                    missing_or_error=sum(r['candidate_status'] not in ('completed','context_unsupported') for r in rs)))
    write_csv(output/'coverage.csv',coverage);return all_rows

def require_complete(rows,haiku=True):
    missing=[r for r in rows if r['candidate_status'] not in ('completed','context_unsupported') or
             (haiku and r.get('haiku_reference_status')!='completed')]
    if missing:raise RuntimeError(f'{len(missing)} candidate/reference records unfinished. Continue generation before freezing judge jobs.')

def consolidate_judgments(root):
    root=Path(root);all_rows=[]
    current={COMMON_JUDGE_RUN,HAIKU_SELF_JUDGE_RUN}
    for directory in (root/'results/api').iterdir():
        if directory.name not in current or not (directory/'responses.csv').exists():continue
        for row in read_csv(directory/'responses.csv'):all_rows.append(dict(row,judge_run=directory.name))
    write_csv(root/'results/compiled/all_judgments.csv',all_rows)
    return all_rows

def intervals(root,draws=2000):
    rows=consolidate_judgments(root)
    source=records(root)
    common=[r for r in rows if r.get('judge_run')==COMMON_JUDGE_RUN and r.get('status')=='completed']
    baseline={(r['model'],r['prompt_id']):r for r in common if r['condition']=='untreated_hf'}
    metrics=sorted({k for r in common for k in r if k.startswith('score_')} |
                   {'judge_prediction_similarity','judge_reasoning_similarity','confidence_label_agreement'})
    output=[]
    def number(v):
        try:
            value=float(v)
            return value if value in (0,1,2,3,4) else None
        except (ValueError,TypeError):return None
    for model in MODELS:
        for condition in CONDITIONS:
            for split in ('validation','test'):
                matched=[(r,baseline.get((model,r['prompt_id']))) for r in common
                         if r['model']==model and r['condition']==condition and r['split']==split]
                for metric in metrics:
                    groups=defaultdict(list);missing=0
                    for r,b in matched:
                        x=number(r.get(metric));y=number(b.get(metric)) if b else None
                        if x is None or y is None:missing+=1;continue
                        if r['prompt_sha256']!=b['prompt_sha256']:raise ValueError('Unmatched prompt texts')
                        groups[(r['scenario'],r['step'])].append(x-y)
                    n=sum(map(len,groups.values()));point=sum(map(sum,groups.values()))/n if n else None
                    low=high=None
                    if len(groups)>=2:
                        rng=random.Random(42);keys=sorted(groups);samples=[]
                        for _ in range(draws):
                            selected=[groups[rng.choice(keys)] for __ in keys]
                            samples.append(sum(map(sum,selected))/sum(map(len,selected)))
                        samples.sort();low=samples[int(.025*draws)];high=samples[int(.975*draws)]
                    expected=sum(r['split']==split for r in source)
                    output.append(dict(model=model,condition=condition,split=split,metric=metric,
                        expected_pairs=expected,matched_rows=n,unavailable_pairs=expected-n,independent_groups=len(groups),
                        sft_minus_untreated=point,low95=low,high95=high,
                        interpretation='exploratory multiple comparisons; CIs not multiplicity-adjusted'))
    write_csv(Path(root)/'results/compiled/grouped_paired_intervals.csv',output);return output
