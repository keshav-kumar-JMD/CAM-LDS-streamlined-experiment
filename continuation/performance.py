"""Bounded training-only runtime tuning for the two untreated TQ1_0 7B runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import time

from streamlined.common import digest, read_json, write_json
from streamlined.data import records
from streamlined.encoding import render
from streamlined.telemetry import Meter
from streamlined.transport import file_hash
from .artifacts import artifact_paths

TARGETS = {'mistral7b', 'qwen7b'}
BOUNDARIES = (4096, 8192, 16384)
VERSION = '7b_tq1_0_buckets_v1'
BENCH_OUTPUT_TOKENS = 32


def applicable(key, representation, adapter):
    return key in TARGETS and representation == 'tq1_0' and adapter is None


def bucket(input_tokens):
    for upper in BOUNDARIES:
        if input_tokens <= upper:
            return str(upper)
    return 'long'


def profile_path(root, key, gpu_arch):
    return Path(root)/'telemetry/performance'/f'{key}_tq1_0_sm{gpu_arch}.json'


def gpu_identity():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Performance pilot requires a CUDA runtime')
    major,minor=torch.cuda.get_device_capability()
    memory_gib=round(torch.cuda.get_device_properties(0).total_memory/(1024**3))
    return f'{major}{minor}_vram{memory_gib}gib',torch.cuda.get_device_name()


def load_profile(root,key,representation,adapter,mirror,artifact_sha):
    if not applicable(key,representation,adapter):return None
    arch,gpu=gpu_identity();path=profile_path(root,key,arch)
    if not path.exists() and mirror:
        mirror.fetch(path.relative_to(root).as_posix(),path)
    if not path.exists():
        raise RuntimeError(f'Run the training-only performance pilot before full {key} inference: {path}')
    value=read_json(path)
    if (value.get('version')!=VERSION or value.get('model')!=key or
            value.get('artifact_sha256')!=artifact_sha or value.get('gpu_arch')!=arch):
        raise ValueError('Frozen performance profile does not match model, GPU architecture or artifact')
    return value


def training_sample(root,key,tokenizer,context):
    eligible={str(x) for x in read_json(Path(root)/'manifests/experiment.json')['eligible_ids']}
    available={name:[] for name in ('4096','8192','16384','long')}
    for row in records(root):
        if row['split']!='train' or str(row['prompt_id']) not in eligible:continue
        prefix,_,_=render(tokenizer,row['prompt_text'],'')
        ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
        if len(ids)+BENCH_OUTPUT_TOKENS>context:continue
        available[bucket(len(ids))].append((row['prompt_id'],ids))
    # At most twenty frozen eligible training IDs; the same IDs are used for every
    # candidate within a length bucket.
    quota={'4096':16,'8192':4,'16384':0,'long':0}
    chosen={}
    for name,items in available.items():
        items.sort(key=lambda item:(len(item[1]),int(item[0])))
        n=min(quota[name],len(items))
        if n:
            indexes=sorted({round(i*(len(items)-1)/max(1,n-1)) for i in range(n)})
            chosen[name]=[items[i] for i in indexes]
    if not chosen:raise RuntimeError('No eligible training prompts fit the native context')
    return chosen


def _run_trial(root,key,artifact,sample,settings,mirror,label):
    # Import here to avoid a module cycle with continuation.infer.
    from .infer import Server
    slots=min(settings['slots'],max(1,len(sample)))
    context=max(len(ids)+BENCH_OUTPUT_TOKENS for _,ids in sample)
    meter=Meter(root,f'benchmark/{key}/{label}')
    start=time.perf_counter();work_seconds=None;startup_seconds=None
    answers={};failure='';server_log=''
    try:
        with Server(root,key,artifact,context,slots,mirror,options=settings) as server:
            startup_seconds=time.perf_counter()-start
            work_start=time.perf_counter()
            server_log=str(server.log.relative_to(root))
            def timed_completion(ids):
                started=time.perf_counter()
                response=server.completion(ids,BENCH_OUTPUT_TOKENS)
                return response,time.perf_counter()-started
            with ThreadPoolExecutor(max_workers=slots) as executor:
                futures={executor.submit(timed_completion,ids):str(pid)
                         for pid,ids in sample}
                for future in as_completed(futures):
                    answer,request_seconds=future.result()
                    if answer.get('truncated'):
                        raise RuntimeError('Performance trial truncated an input')
                    answers[futures[future]]={'content':answer.get('content',''),
                        'hash':digest(answer.get('content','')),
                        'tokens':len(answer.get('tokens',[])),
                        'timings':answer.get('timings',{}),
                        'request_wall_seconds':request_seconds}
            work_seconds=time.perf_counter()-work_start
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
    metrics=meter.close();seconds=time.perf_counter()-start
    output=sum(x['tokens'] for x in answers.values())
    prompt=sum(len(ids) for _,ids in sample)
    report={'name':label,'settings':settings,'prompt_ids':[str(x[0]) for x in sample],
        'wall_seconds':seconds,'server_startup_seconds':startup_seconds,
        'request_work_seconds':work_seconds,'prompt_tokens':prompt,'output_tokens':output,
        'aggregate_prompt_tokens_per_second':prompt/work_seconds if work_seconds else 0,
        'aggregate_output_tokens_per_second':output/work_seconds if work_seconds else 0,
        'approximate_slot_busy_fraction':min(1,sum(x['request_wall_seconds'] for x in answers.values())/
            (slots*work_seconds)) if work_seconds else None,
        'mean_gpu_utilization_pct':metrics.get('mean_gpu_utilization_pct'),
        'sampled_peak_gpu_memory_bytes':metrics.get('sampled_peak_gpu_memory_bytes'),
        'estimated_energy_wh':metrics.get('estimated_energy_wh'),
        'answers':answers,'server_log':server_log,'error':failure}
    return report


def _audit_backend(root,artifact,trials,arch):
    from .artifacts import toolchain
    toolkit=toolchain(root,bitnet=False,mirror=None)
    build=toolkit['repo']/'build'
    cache=build/'CMakeCache.txt'
    cmake_text=cache.read_text(errors='replace') if cache.exists() else ''
    marker=build/'camlds_build.json'
    build_info=read_json(marker) if marker.exists() else {}
    logs=[]
    for trial in trials:
        path=Path(root)/trial['server_log'] if trial['server_log'] else None
        if path and path.exists():logs.append(path.read_text(errors='replace')[:40000])
    startup='\n'.join(logs)
    cuda_evidence=bool(re.search(r'ggml_cuda|CUDA0|offload.*GPU',startup,re.I))
    layer_matches=[(int(a),int(b)) for a,b in
                   re.findall(r'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers',startup,re.I)]
    return {'gpu_offload_requested_layers':999,
        'cuda_build_enabled':'GGML_CUDA:BOOL=ON' in cmake_text or 'GGML_CUDA:UNINITIALIZED=ON' in cmake_text,
        'release_build_recorded':'CMAKE_BUILD_TYPE:STRING=Release' in cmake_text,
        'build_cuda_architecture':build_info.get('cuda_arch'),
        'build_cuda_arch_matches_runtime':build_info.get('cuda_arch')==arch.split('_')[0],
        'cuda_startup_log_evidence':cuda_evidence,
        'observed_layer_offload':layer_matches,
        'all_observed_layers_offloaded':bool(layer_matches) and all(a==b for a,b in layer_matches),
        'cuda_graph_log_evidence':bool(re.search(r'CUDA graph|cuda_graph',startup,re.I)),
        'mmq_build_flag':next((line for line in cmake_text.splitlines()
            if line.startswith('GGML_CUDA_FORCE_MMQ:')), 'not recorded'),
        'mmq_kernel_confirmed':bool(re.search(r'MMQ.*(?:enabled|kernel)|(?:enabled|kernel).*MMQ',startup,re.I)),
        'interpretation':'CUDA utilization/memory are measured; per-operation MMQ dispatch is unverified unless the startup log explicitly identifies it'}


def benchmark(root,key,representation,mirror):
    if not applicable(key,representation,None):raise ValueError('Only untreated Mistral/Qwen 7B TQ1_0')
    from .infer import load_tokenizer
    root=Path(root);folder,artifact,manifest=artifact_paths(root,key,representation)
    if not manifest.exists():mirror.fetch(manifest.relative_to(root).as_posix(),manifest)
    if not artifact.exists():mirror.fetch(artifact.relative_to(root).as_posix(),artifact)
    saved=read_json(manifest)
    if file_hash(artifact)!=saved['artifact_sha256']:raise ValueError('GGUF artifact checksum mismatch')
    arch,gpu=gpu_identity();path=profile_path(root,key,arch)
    if not path.exists():mirror.fetch(path.relative_to(root).as_posix(),path)
    if path.exists():
        current=read_json(path)
        if (current.get('version')==VERSION and current.get('artifact_sha256')==saved['artifact_sha256']):
            print('Reusing frozen performance profile:',path,flush=True)
            return current
        raise ValueError('Existing performance profile belongs to a different artifact or revision')
    tokenizer=load_tokenizer(root,key)
    native_context=int(read_json(root/'manifests/models.json')[key]['context'])
    samples=training_sample(root,key,tokenizer,native_context)
    results={};selected={};all_trials=[]
    for name,items in samples.items():
        # Give every trial the same prompt IDs and response allowance.
        trials=[]
        base={'slots':min(2,len(items)),'batch':2048,'ubatch':512,
              'flash_attn':'default','threads':None,'verbosity':3}
        def trial(label,settings):
            trial=_run_trial(root,key,artifact,items,settings,mirror,name+'_'+label)
            trials.append(trial)
            print(name,label,'tokens/s',round(trial['aggregate_output_tokens_per_second'],2),
                  'error',trial['error'] or 'none',flush=True)
            baseline=trials[0]['answers']
            matched=(set(trial['answers'])==set(baseline) and
                     all(trial['answers'][pid]['hash']==baseline[pid]['hash'] for pid in baseline))
            trial['pilot_response_hashes_match_baseline']=matched
            return trial
        first=trial('baseline',base)
        if first['error'] or len(first['answers'])!=len(items):
            raise RuntimeError(f'Baseline performance pilot failed in bucket {name}: {first["error"]}')
        best=first
        for label,settings in [('batch',dict(base,batch=4096,ubatch=1024)),
                               ('flash',dict(base,batch=4096,ubatch=1024,flash_attn='on'))]:
            candidate=trial(label,settings)
            if (not candidate['error'] and candidate['pilot_response_hashes_match_baseline'] and
                    candidate['aggregate_output_tokens_per_second']>best['aggregate_output_tokens_per_second']):
                best=candidate
        previous_slot=best
        for slots in (4,8,16):
            if len(items)<slots:break
            candidate=trial('slots'+str(slots),dict(best['settings'],slots=slots))
            if (not candidate['error'] and candidate['pilot_response_hashes_match_baseline'] and
                    candidate['aggregate_output_tokens_per_second']>best['aggregate_output_tokens_per_second']):
                best=candidate
            if slots==8 and candidate['aggregate_output_tokens_per_second']<=previous_slot['aggregate_output_tokens_per_second']:
                break
            previous_slot=candidate
        for threads in (4,8,12):
            candidate=trial('threads'+str(threads),dict(best['settings'],threads=threads))
            if (not candidate['error'] and candidate['pilot_response_hashes_match_baseline'] and
                    candidate['aggregate_output_tokens_per_second']>best['aggregate_output_tokens_per_second']):
                best=candidate
        selected[name]=dict(best['settings'],verbosity=2,bucket=name,
            benchmark_prompt_ids=[str(pid) for pid,_ in items],
            benchmark_aggregate_output_tokens_per_second=best['aggregate_output_tokens_per_second'])
        results[name]=trials;all_trials.extend(trials)
    # Some buckets have no training examples. They still need a defined runtime
    # setting so that validation/test IDs cannot silently disappear.
    for name in ('4096','8192','16384','long'):
        if name not in selected:
            selected[name]=dict(slots=2,batch=2048,ubatch=512,
                flash_attn='default',threads=None,verbosity=2,bucket=name,
                benchmark_prompt_ids=[],benchmark_aggregate_output_tokens_per_second=None,
                unbenchmarked_bucket=True,uses_original_backend_defaults=True)
    value={'version':VERSION,'model':key,'representation':representation,
           'artifact_sha256':saved['artifact_sha256'],'gpu_arch':arch,'gpu':gpu,
           'native_context':native_context,'benchmark_output_tokens':BENCH_OUTPUT_TOKENS,
           'training_only':True,'prompt_caching':False,
           'method':'fixed eligible training IDs; select fastest hash-matched trial',
           'buckets':selected,'trials':results,
           'backend_audit':_audit_backend(root,artifact,all_trials,arch)}
    value['profile_sha256']=digest({k:v for k,v in value.items() if k!='profile_sha256'})
    write_json(path,value);mirror.put(root,path)
    print('Frozen performance profile:',path,flush=True)
    return value
