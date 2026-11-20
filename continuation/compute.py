"""Auditable per-stream compute accounting for notebooks 00--07."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import shutil

from streamlined.common import read_json, write_csv, verify_experiment


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, default=0.0):
    try:
        value=float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError): return default


def _safe(value): return re.sub(r'[^a-z0-9]+','_',str(value).lower()).strip('_') or 'unknown'


def _notebook_prefix(condition,representation,phase=''):
    """Notebook that produced the model/telemetry stream represented by a file."""
    if phase=='training': return '01'
    if representation=='hf': return '02' if condition=='untreated_hf' else '01'
    if representation=='iq1_s_camlds_calibrated': return '06'
    if representation in ('tq1_0','native_i2_s'): return '04' if condition=='untreated' else '05'
    return '09'


def _group(rows,fields):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row.get(field,'') for field in fields)].append(row)
    return groups


def _sample_integrals(data):
    samples=data.get('samples') or [];seconds=weighted=previous=0.0
    for sample in samples:
        elapsed=_number(sample.get('elapsed_seconds'),previous+2.0)
        interval=max(0.0,elapsed-previous);seconds+=interval
        weighted+=interval*_number(sample.get('gpu_utilization_pct'));previous=elapsed
    if not seconds and data.get('mean_gpu_utilization_pct') is not None:
        seconds=_number(data.get('wall_seconds'));weighted=seconds*_number(data.get('mean_gpu_utilization_pct'))
    return seconds,weighted


def _telemetry_identity(root,path,data):
    parts=path.relative_to(root/'telemetry').parts;run=str(data.get('run',''))
    if parts[0]=='quantized':
        model,condition,representation=parts[1:4]
        phase='quantized_preparation' if 'prepare' in parts[4:] or run.endswith('/prepare') else 'quantized_inference'
        return model,condition,representation,phase,('all' if condition=='untreated' else 'validation_test')
    if parts[0] in ('performance','billing'): return '',parts[0],'',parts[0],''
    model=parts[0]
    if len(parts)>1 and parts[1]=='train': return model,'sft_'+parts[2],'hf','training','train'
    condition=parts[1] if len(parts)>1 else ''
    return model,('untreated_hf' if condition=='untreated_hf' else 'sft_'+condition),'hf','hf_inference',(parts[2] if len(parts)>2 else '')


def _inventory(root):
    rows=[]
    for path in sorted((root/'telemetry').rglob('*.json')):
        relative=path.relative_to(root).as_posix()
        if 'tokenizer_audit' in path.name or '/billing/' in '/'+relative: continue
        try: data=read_json(path)
        except Exception: continue
        if not isinstance(data,dict) or 'wall_seconds' not in data: continue
        model,condition,representation,phase,split=_telemetry_identity(root,path,data)
        if phase=='performance': continue
        sampled_seconds,util_seconds=_sample_integrals(data);status=str(data.get('status','unknown'))
        rows.append(dict(
            telemetry_path=relative,telemetry_sha256=_sha(path),invocation_id=path.stem,run=data.get('run',''),
            model=model,condition=condition,representation=representation,phase=phase,split=split,status=status,
            invocation_kind=('pilot' if 'pilot' in status or 'pilot' in str(data.get('run','')) else 'experiment'),
            config_hash=data.get('config_hash',''),runtime_revision=data.get('runtime_revision',''),gpu=data.get('gpu',''),
            cuda=data.get('cuda',''),nvidia_smi=data.get('nvidia_smi',''),wall_seconds=_number(data.get('wall_seconds')),
            model_load_seconds=_number(data.get('model_load_seconds')),
            model_cache_restore_download_sync_seconds=_number(data.get('model_cache_restore_download_sync_seconds')),
            training_wall_seconds=_number(data.get('training_wall_seconds_including_checkpoint_transfers')),
            preparation_and_upload_wall_seconds=_number(data.get('preparation_and_upload_wall_seconds')),
            inference_and_io_wall_seconds=_number(data.get('inference_and_io_wall_seconds',data.get('inference_seconds'))),
            generation_seconds=_number(data.get('generation_seconds')),
            local_checkpoint_seconds=_number(data.get('local_checkpoint_seconds')),
            receiver_seconds=_number(data.get('receiver_seconds')),csv_snapshot_seconds=_number(data.get('csv_snapshot_seconds')),
            input_tokens=_number(data.get('input_tokens',data.get('session_input_tokens'))),
            output_tokens=_number(data.get('output_tokens',data.get('session_output_tokens'))),new_rows=_number(data.get('new_rows')),
            sampled_seconds=sampled_seconds,gpu_utilization_weighted_seconds=util_seconds,
            mean_gpu_utilization_pct=_number(data.get('mean_gpu_utilization_pct')),
            sampled_peak_gpu_memory_bytes=_number(data.get('sampled_peak_gpu_memory_bytes')),
            peak_allocated_bytes=_number(data.get('peak_allocated_bytes')),peak_reserved_bytes=_number(data.get('peak_reserved_bytes')),
            sampled_peak_server_rss_bytes=_number(data.get('sampled_peak_server_rss_bytes')),
            estimated_energy_wh=_number(data.get('estimated_energy_wh')),parallel_slots=data.get('parallel_slots',''),
            backend=data.get('backend',''),artifact_transfer_accounting=('included_in_preparation_phase' if phase=='quantized_preparation' else 'not_separately_instrumented_or_not_applicable')))
    hashes=defaultdict(int)
    for row in rows: hashes[row['telemetry_sha256']]+=1
    for row in rows: row['duplicate_content_hash_count']=hashes[row['telemetry_sha256']]
    return rows


def _record_identity(root,path,row):
    parts=path.relative_to(root/'results').parts
    if parts[0]=='quantized': return row.get('model',parts[1]),row.get('condition',parts[2]),row.get('representation',parts[3])
    condition=row.get('condition',parts[1])
    return row.get('model',parts[0]),('untreated_hf' if condition=='untreated_hf' else 'sft_'+condition.removeprefix('sft_')),'hf'


def _row_performance(root):
    rows=[]
    for path in sorted((root/'results').rglob('records/*.json')):
        relative=path.relative_to(root).as_posix()
        if '/api/' in '/'+relative: continue
        try: row=read_json(path)
        except Exception: continue
        if not isinstance(row,dict) or 'prompt_id' not in row or 'model' not in row: continue
        model,condition,representation=_record_identity(root,path,row);attempts=max(1,int(_number(row.get('attempt_count'),1)))
        elapsed=_number(row.get('elapsed_seconds'));output=_number(row.get('output_tokens'))
        rows.append(dict(model=model,condition=condition,representation=representation,split=row.get('split',''),
            prompt_id=row.get('prompt_id',''),status=row.get('status',''),attempt_count=attempts,retry_count=max(0,attempts-1),
            input_tokens=_number(row.get('input_tokens')),output_tokens=output,elapsed_seconds=elapsed,
            output_tokens_per_second=(output/elapsed if elapsed else 0),finish_reason=row.get('finish_reason',''),
            generation_error=row.get('generation_error',''),input_truncated=row.get('input_truncated',''),
            response_truncated=row.get('response_truncated',''),output_allowance_reduced=row.get('output_allowance_reduced',''),
            max_new_tokens=row.get('max_new_tokens',''),context=row.get('context',''),parallel_slots=row.get('parallel_slots',''),
            runtime_revision=row.get('runtime_revision',''),peak_allocated_bytes=_number(row.get('peak_allocated_bytes')),
            peak_reserved_bytes=_number(row.get('peak_reserved_bytes')),record_path=relative,
            record_bytes=path.stat().st_size,record_modified_ns=path.stat().st_mtime_ns))
    return rows


def _percentile(values,p):
    values=sorted(values)
    if not values:return 0
    position=(len(values)-1)*p;low=int(position);high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


def _sum(rows,key): return sum(_number(r.get(key)) for r in rows)


def _stream_telemetry(rows):
    output=[]
    for key,items in sorted(_group(rows,('model','condition','representation','phase')).items()):
        experiment=[r for r in items if r['invocation_kind']=='experiment'];pilot=[r for r in items if r['invocation_kind']=='pilot']
        sampled=_sum(experiment,'sampled_seconds');util=_sum(experiment,'gpu_utilization_weighted_seconds')
        wall=_sum(experiment,'wall_seconds');outputs=_sum(experiment,'output_tokens')
        output.append(dict(model=key[0],condition=key[1],representation=key[2],phase=key[3],
            aggregation_scope='exact_stream_resource_sum_including_retries_and_resumes',invocation_count=len(experiment),
            pilot_invocation_count=len(pilot),completed_invocations=sum(r['status']=='completed' for r in experiment),
            noncompleted_invocations=sum(r['status']!='completed' for r in experiment),experimental_resource_wall_seconds=wall,
            experimental_resource_gpu_hours=wall/3600,pilot_resource_wall_seconds=_sum(pilot,'wall_seconds'),
            model_load_seconds=_sum(experiment,'model_load_seconds'),model_cache_restore_download_sync_seconds=_sum(experiment,'model_cache_restore_download_sync_seconds'),
            training_wall_seconds=_sum(experiment,'training_wall_seconds'),preparation_and_upload_wall_seconds=_sum(experiment,'preparation_and_upload_wall_seconds'),
            inference_and_io_wall_seconds=_sum(experiment,'inference_and_io_wall_seconds'),generation_seconds=_sum(experiment,'generation_seconds'),
            local_checkpoint_seconds=_sum(experiment,'local_checkpoint_seconds'),receiver_seconds=_sum(experiment,'receiver_seconds'),
            csv_snapshot_seconds=_sum(experiment,'csv_snapshot_seconds'),input_tokens=_sum(experiment,'input_tokens'),
            output_tokens=outputs,new_rows_reported=_sum(experiment,'new_rows'),
            operational_output_tokens_per_resource_second=(outputs/wall if wall else 0),sampled_gpu_seconds=sampled,
            time_weighted_mean_gpu_utilization_pct=(util/sampled if sampled else 0),
            sampled_peak_gpu_memory_bytes=max((_number(r['sampled_peak_gpu_memory_bytes']) for r in experiment),default=0),
            peak_allocated_bytes=max((_number(r['peak_allocated_bytes']) for r in experiment),default=0),
            peak_reserved_bytes=max((_number(r['peak_reserved_bytes']) for r in experiment),default=0),
            sampled_peak_server_rss_bytes=max((_number(r['sampled_peak_server_rss_bytes']) for r in experiment),default=0),
            estimated_energy_wh=_sum(experiment,'estimated_energy_wh'),
            artifact_transfer_accounting='separate preparation phase where instrumented; never assigned to inference'))
    return output


def _coverage(rows):
    output=[]
    for key,items in sorted(_group(rows,('model','condition','representation')).items()):
        statuses=defaultdict(int)
        for row in items: statuses[row['status']]+=1
        elapsed=[_number(r['elapsed_seconds']) for r in items if r['status']=='completed'];seconds=sum(elapsed)
        output_tokens=sum(_number(r['output_tokens']) for r in items if r['status']=='completed')
        output.append(dict(model=key[0],condition=key[1],representation=key[2],record_count=len(items),
            completed=statuses['completed'],context_unsupported=statuses['context_unsupported'],retryable_error=statuses['retryable_error'],
            other_status=sum(statuses.values())-statuses['completed']-statuses['context_unsupported']-statuses['retryable_error'],
            total_attempts=sum(int(r['attempt_count']) for r in items),total_retries=sum(int(r['retry_count']) for r in items),
            completed_input_tokens=sum(_number(r['input_tokens']) for r in items if r['status']=='completed'),
            completed_output_tokens=output_tokens,completed_elapsed_seconds=seconds,
            completed_output_tokens_per_request_work_second=(output_tokens/seconds if seconds else 0),
            elapsed_p25_seconds=_percentile(elapsed,.25),elapsed_median_seconds=_percentile(elapsed,.5),
            elapsed_p75_seconds=_percentile(elapsed,.75),elapsed_p95_seconds=_percentile(elapsed,.95),
            runtime_revisions='|'.join(sorted({str(r['runtime_revision']) for r in items if r['runtime_revision']}))))
    return output


def _model_summary(streams):
    result=[]
    for key,items in sorted(_group(streams,('model','representation','phase')).items()):
        seconds=_sum(items,'experimental_resource_wall_seconds')
        result.append(dict(model=key[0],representation=key[1],phase=key[2],
            aggregation_scope='model_resource_total_across_distinct_streams_not_calendar_elapsed_time',stream_count=len(items),
            invocation_count=sum(int(r['invocation_count']) for r in items),resource_wall_seconds=seconds,resource_gpu_hours=seconds/3600,
            input_tokens=_sum(items,'input_tokens'),output_tokens=_sum(items,'output_tokens'),estimated_energy_wh=_sum(items,'estimated_energy_wh'),
            peak_gpu_memory_bytes=max((_number(r['sampled_peak_gpu_memory_bytes']) for r in items),default=0)))
    return result


def _api_usage(root):
    groups=defaultdict(list)
    for path in sorted((root/'results/api').glob('*/records/*.json')):
        try: row=read_json(path)
        except Exception: continue
        if row.get('status')=='completed': groups[(path.parent.parent.name,row.get('model',''),row.get('condition',''),row.get('representation',''))].append(row)
    output=[]
    for key,items in sorted(groups.items()):
        def usage(name): return sum(_number((r.get('usage') or {}).get(name,r.get(name))) for r in items)
        output.append(dict(api_run=key[0],model=key[1],condition=key[2],representation=key[3],completed_requests=len(items),
            input_tokens=usage('input_tokens'),output_tokens=usage('output_tokens'),cache_creation_input_tokens=usage('cache_creation_input_tokens'),
            cache_read_input_tokens=usage('cache_read_input_tokens'),estimated_cost_usd=sum(_number(r.get('estimated_cost_usd')) for r in items)))
    return output


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8');temporary.replace(path)


def summarize(root):
    root=Path(root);experiment_manifest=verify_experiment(root);output=root/'results/final';output.mkdir(parents=True,exist_ok=True)
    # Remove the legacy flat compute namespace, then rebuild the categorized
    # hierarchy atomically from authoritative telemetry/row records.
    for path in output.glob('compute__*'):
        if path.is_file(): path.unlink()
    legacy_manifest=output/'manifest__compute_files.csv'
    if legacy_manifest.is_file(): legacy_manifest.unlink()
    compute_root=output/'compute'
    if compute_root.exists(): shutil.rmtree(compute_root)
    invocation_output=compute_root/'invocations';row_output=compute_root/'rows';summary_output=compute_root/'stream_summary'
    for folder in (invocation_output,row_output,summary_output): folder.mkdir(parents=True,exist_ok=True)
    inventory=_inventory(root);row_performance=_row_performance(root);stream_summary=_stream_telemetry(inventory);coverage=_coverage(row_performance)
    coverage_index={(r['model'],r['condition'],r['representation']):r for r in coverage}
    for row in stream_summary:
        match=coverage_index.get((row['model'],row['condition'],row['representation']),{})
        for field in ('record_count','completed','context_unsupported','retryable_error','other_status','total_attempts','total_retries','runtime_revisions'): row[field]=match.get(field,'')
    outputs={invocation_output/'09__compute__invocation_inventory.csv':inventory,
        row_output/'09__compute__row_performance.csv':row_performance,
        row_output/'09__compute__coverage_and_retries.csv':coverage,
        summary_output/'09__compute__stream_summary.csv':stream_summary,
        summary_output/'09__compute__model_resource_summary.csv':_model_summary(stream_summary),
        summary_output/'09__compute__api_usage_summary.csv':_api_usage(root)}
    for path,rows in outputs.items(): write_csv(path,rows)
    for key,items in sorted(_group(inventory,('model','condition','representation','phase')).items()):
        prefix=_notebook_prefix(key[1],key[2],key[3])
        name=prefix+'__compute__invocations__'+'__'.join(_safe(v) for v in key)+'.csv';path=invocation_output/name;write_csv(path,items);outputs[path]=items
    for key,items in sorted(_group(row_performance,('model','condition','representation')).items()):
        prefix=_notebook_prefix(key[1],key[2])
        name=prefix+'__compute__rows__'+'__'.join(_safe(v) for v in key)+'.csv';path=row_output/name;write_csv(path,items);outputs[path]=items
    for row in stream_summary:
        key=(row['model'],row['condition'],row['representation'],row['phase'])
        prefix=_notebook_prefix(key[1],key[2],key[3])
        name=prefix+'__compute__stream_summary__'+'__'.join(_safe(v) for v in key)+'.csv'
        path=summary_output/name;write_csv(path,[row]);outputs[path]=[row]
    dictionary=[
        dict(output='09__compute__invocation_inventory.csv',meaning='One row per telemetry invocation; retries/resumes remain separate.'),
        dict(output='09__compute__stream_summary.csv',meaning='Resource sums within exact model/condition/representation/phase only.'),
        dict(output='09__compute__model_resource_summary.csv',meaning='Model resource totals by representation and phase; not project elapsed time.'),
        dict(output='09__compute__coverage_and_retries.csv',meaning='Terminal status, attempts and retries from response records.'),
        dict(output='09__compute__row_performance.csv',meaning='Per-response timing/token audit without prompt or response text.'),
        dict(output='09__compute__api_usage_summary.csv',meaning='Haiku token/cache/cost usage where recorded; excludes Colab billing.'),
        dict(output='01--06__compute__invocations__*.csv',meaning='Source-notebook-prefixed per-stream/phase invocation audit files.'),
        dict(output='01--06__compute__rows__*.csv',meaning='Source-notebook-prefixed per-stream response performance files.'),
        dict(output='01--06__compute__stream_summary__*.csv',meaning='Source-notebook-prefixed one-row report for each exact stream and phase.')]
    dictionary_path=summary_output/'09__compute__data_dictionary.csv';write_csv(dictionary_path,dictionary);outputs[dictionary_path]=dictionary
    source_paths=sorted({r['telemetry_path'] for r in inventory}|{r['record_path'] for r in row_performance})
    _write_json(compute_root/'09__compute__aggregation_manifest.json',dict(config_hash=experiment_manifest['config_hash'],
        policy='exact-stream resource accounting; pilots separate; preparation never assigned to inference; no Colab billing',
        telemetry_invocations=len(inventory),response_records=len(row_performance),source_file_count=len(source_paths),source_paths=source_paths))
    file_rows=[]
    for path in sorted(compute_root.rglob('*__compute__*')):
        count=''
        if path.suffix=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as handle: count=max(0,sum(1 for _ in csv.reader(handle))-1)
        file_rows.append(dict(file=path.relative_to(output).as_posix(),rows=count,sha256=_sha(path)))
    manifest_path=compute_root/'09__manifest__compute_files.csv';write_csv(manifest_path,file_rows)
    return dict(output_folder=str(output),telemetry_invocations=len(inventory),response_records=len(row_performance),
        exact_stream_phases=len(stream_summary),compute_files=len(file_rows)+1,manifest=str(manifest_path))
