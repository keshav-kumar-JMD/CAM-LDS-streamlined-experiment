"""One explicit final result namespace for notebooks 00--07."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import os
import shutil
from streamlined.common import CONDITIONS, MODELS, digest, read_csv, read_json, write_csv, verify_experiment
from streamlined.data import records
from streamlined.transport import file_hash
from .judge import expected_streams
from .protocol import NEW_JUDGE_RUN, OPTIONAL_JUDGE_RUN, stream

def _records(folder):
    return [read_json(p) for p in sorted((Path(folder)/'records').glob('*.json'))]

def _write(root,output,name,rows,expected,source,kind,manifest,allow_incomplete=False):
    rows=list(rows)
    ids=[str(r['prompt_id']) for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError(f'{name}: repeated prompt ID')
    actual=len(rows)
    terminal=sum(r.get('status') in ('completed','context_unsupported') for r in rows)
    complete=actual==expected and terminal==expected
    if complete or allow_incomplete:
        path=output/name
        write_csv(path,sorted(rows,key=lambda r:int(r['prompt_id'])))
        sha=digest(path.read_bytes())
        exported=True
    else:
        sha='';exported=False
    manifest.append(dict(file=name,kind=kind,source=source,expected=expected,
        records=actual,completed=sum(r.get('status')=='completed' for r in rows),
        context_unsupported=sum(r.get('status')=='context_unsupported' for r in rows),
        other=actual-terminal,missing=expected-actual,
        complete=complete,exported=exported,sha256=sha))

def _joined(root,model,condition):
    rows=[]
    for split in ('validation','test'):
        part=_records(root/'results'/model/condition/split)
        if any(r.get('split')!=split for r in part):raise ValueError(f'{model}/{condition}/{split}: wrong split')
        rows.extend(part)
    return rows

def _judge_groups(folder):
    groups={}
    folder=Path(folder)
    snapshot=folder/'responses.csv'
    record_paths=list((folder/'records').glob('*.json'))
    if (snapshot.exists() and
            all(snapshot.stat().st_mtime_ns>=p.stat().st_mtime_ns for p in record_paths)):
        rows=read_csv(snapshot)
        if len(rows)!=len(record_paths):
            raise ValueError(f'{folder}: CSV/record count differs; refresh snapshot before final export')
    else:
        rows=[read_json(p) for p in record_paths]
    for row in rows:
        key=(row.get('model',''),row.get('condition',''),row.get('representation',''))
        groups.setdefault(key,[]).append(row)
    return groups

def consolidate(root,include_optional=True):
    root=Path(root);verify_experiment(root)
    output=root/'results/final';output.mkdir(parents=True,exist_ok=True)
    legacy=output/'file_manifest.csv'
    if legacy.exists():legacy.unlink()
    source={str(r['prompt_id']):r for r in records(root)}
    manifest=[]
    def check(rows,name):
        for row in rows:
            origin=source.get(str(row['prompt_id']))
            if origin is None or row.get('prompt_sha256')!=origin['prompt_sha256']:
                raise ValueError(f'{name}: prompt identity mismatch')
    # Source/reference evidence is retained verbatim and named separately.
    for filename,kind in (('prompts.csv','prompt'),('teacher_responses.csv','teacher'),
                          ('split_manifest.csv','split'),('ground_truth.csv','truth')):
        rows=read_csv(root/'data'/filename)
        path=output/('source__'+filename)
        write_csv(path,rows)
        manifest.append(dict(file=path.name,kind=kind,source='data/'+filename,
            expected=990,records=len(rows),completed=len(rows),context_unsupported=0,
            other=0,missing=990-len(rows),complete=len(rows)==990,exported=True,
            sha256=digest(path.read_bytes())))
    # Notebook 03's already compiled cross-model tables are complete standalone
    # products. Preserve their exact bytes; do not reserialize nested JSON cells.
    compiled=root/'results/compiled'
    for name in ('D_all_responses.csv','E_technique_validation.csv',
                 'E_explanation_validation.csv','E_joint_validation.csv',
                 'coverage.csv','all_judgments.csv','grouped_paired_intervals.csv'):
        origin=compiled/name
        if not origin.is_file():continue
        destination=output/('analysis__'+name)
        temporary=destination.with_suffix('.partial')
        shutil.copyfile(origin,temporary);os.replace(temporary,destination)
        csv.field_size_limit(100_000_000)
        with origin.open(encoding='utf-8-sig',newline='') as handle:
            count=sum(1 for _ in csv.reader(handle))-1
        manifest.append(dict(file=destination.name,kind='analysis',source=f'results/compiled/{name}',
            expected=count,records=count,completed=count,context_unsupported=0,
            other=0,missing=0,complete=True,exported=True,sha256=file_hash(destination)))
    haiku=_records(root/'results/api/haiku')
    check(haiku,'haiku reference')
    _write(root,output,'reference__haiku__all.csv',haiku,990,'results/api/haiku',
           'reference',manifest,allow_incomplete=True)
    # 01--02, paired validation/test SFT exports; all 990 for untreated HF.
    for model in MODELS:
        rows=_records(root/'results'/model/'untreated_hf/all')
        check(rows,model+'/untreated_hf')
        _write(root,output,f'response__{model}__untreated_hf__all.csv',rows,990,
            f'results/{model}/untreated_hf/all','candidate',manifest)
        for adapter in CONDITIONS:
            rows=_joined(root,model,adapter)
            check(rows,model+'/'+adapter)
            _write(root,output,f'response__{model}__sft_{adapter}__hf__validation_test.csv',
                rows,285,f'results/{model}/{adapter}/validation+test','candidate',manifest)
    # 04--06. A/B are required; C is optional and exported only if complete.
    for family in ('required','optional'):
        if family=='optional' and not include_optional:continue
        for model,representation,adapter,name in expected_streams(family):
            rows=_records(root/'results'/name)
            check(rows,name)
            condition='untreated' if adapter is None else 'sft_'+adapter
            extent='all' if adapter is None and family=='required' else 'validation_test'
            _write(root,output,f'response__{model}__{condition}__{representation}__{extent}.csv',
                rows,990 if extent=='all' else 285,f'results/{name}',
                'candidate',manifest)
    # Judge results include status and raw Haiku messages; incomplete legacy
    # ledgers remain visibly incomplete instead of silently disappearing.
    for run in ('common_uniform_v2','haiku_self_uniform_v2',NEW_JUDGE_RUN,OPTIONAL_JUDGE_RUN):
        folder=root/'results/api'/run
        if not folder.exists():continue
        groups=_judge_groups(folder)
        if run=='haiku_self_uniform_v2':
            rows=[r for group in groups.values() for r in group]
            check(rows,run)
            _write(root,output,'judgment__haiku_self__validation_test.csv',rows,285,
                   f'results/api/{run}','judgment',manifest,allow_incomplete=True)
            continue
        for (model,condition,representation),rows in groups.items():
            check(rows,run+'/'+model+'/'+condition)
            if run=='common_uniform_v2':
                representation='hf';condition='untreated' if condition=='untreated_hf' else 'sft_'+condition
            extent='all' if condition=='untreated' and representation=='hf' else 'validation_test'
            if run==NEW_JUDGE_RUN and condition=='untreated':extent='all'
            tag='legacy_common' if run=='common_uniform_v2' else 'quantized_common'
            if run==OPTIONAL_JUDGE_RUN:tag='quantized_optional_common'
            expected=990 if extent=='all' else 285
            # Judge requests exist only for supported candidate responses.
            if run=='common_uniform_v2':
                candidates=(_records(root/'results'/model/'untreated_hf/all') if condition=='untreated'
                            else _joined(root,model,condition.removeprefix('sft_')))
                expected=sum(r.get('status')=='completed' for r in candidates)
            elif run in (NEW_JUDGE_RUN,OPTIONAL_JUDGE_RUN):
                candidate_stream=stream(model,representation,
                    None if condition=='untreated' else condition.removeprefix('sft_'))
                expected=sum(r.get('status')=='completed' for r in _records(root/'results'/candidate_stream))
            _write(root,output,f'judgment__{tag}__{model}__{condition}__{representation}__{extent}.csv',
                rows,expected,f'results/api/{run}','judgment',manifest,allow_incomplete=True)
    write_csv(output/'manifest__files.csv',manifest)
    tally=dict(files_exported=sum(r['exported'] for r in manifest),
        complete_files=sum(r['complete'] for r in manifest),
        incomplete_files=sum(r['exported'] and not r['complete'] for r in manifest),
        expected_required_candidate_files=60,
        expected_optional_candidate_files=25,
        final_folder=str(output))
    return tally
