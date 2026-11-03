"""Independent, resumable GGUF inference over frozen CAM-LDS prompt IDs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
from streamlined.common import POLICY, audit_response, digest, read_json, verify_experiment, write_json
from streamlined.encoding import render
from streamlined.monitor import Heartbeat
from streamlined.telemetry import Meter
from streamlined.transport import Mirror
from .artifacts import artifact_paths, prepare, toolchain
from .protocol import (PARALLEL_SLOTS, UNTREATED_TQ1_0_7B_RUNTIME_REVISION,
                       UNTREATED_TQ1_0_7B_SLOTS, UPLOAD_BACKPRESSURE,
                       UPLOAD_BATCH_SIZE, identity, selected_rows, stream)
from .store import SyncStore

BITNET_I2S_CONTEXT_RESERVE = 8

def backend_context_reserve(key,representation):
    """Small native-I2_S slot reserve required by the pinned BitNet backend."""
    return BITNET_I2S_CONTEXT_RESERVE if key=='bitnet2b' and representation=='native_i2_s' else 0

def request(url,path,payload=None,timeout=60):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.loads(response.read())

class Server:
    def __init__(self,root,key,artifact,context,parallel,mirror,options=None):
        self.root=Path(root);self.key=key;self.artifact=Path(artifact);self.context=context
        self.parallel=parallel;self.options=options or {}
        self.toolkit=toolchain(root,bitnet=key=='bitnet2b',mirror=mirror)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{self.port}'
        self.log=self.root/'telemetry/logs'/f'gguf_server_{key}_{self.port}.log'
        self.process=None;self.handle=None
    def __enter__(self):
        self.log.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.log.open('a',encoding='utf-8')
        args=[self.toolkit['server'],'-m',self.artifact,'--host','127.0.0.1',
              '--port',str(self.port),'-c',str(self.context*self.parallel),'-ngl','999',
              '--parallel',str(self.parallel),'-b',str(self.options.get('batch',2048)),
              '-ub',str(self.options.get('ubatch',512))]
        if self.options:
            if self.options.get('flash_attn')!='default':
                args.extend(['--flash-attn',self.options['flash_attn']])
            if self.options.get('threads') is not None:
                args.extend(['--threads',str(self.options['threads'])])
            args.extend(['-lv',str(self.options.get('verbosity',2))])
        self.handle.write('COMMAND '+json.dumps([str(x) for x in args])+'\n');self.handle.flush()
        self.process=subprocess.Popen([str(x) for x in args],cwd=self.root,
            stdout=self.handle,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+360
        while time.monotonic()<deadline:
            if self.process.poll() is not None:break
            try:
                health=request(self.url,'/health',timeout=3)
                if health.get('status')=='ok':return self
            except Exception:pass
            time.sleep(1)
        self.__exit__(None,None,None)
        raise RuntimeError(f'llama-server did not become ready; inspect {self.log}')
    def __exit__(self,*_):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.handle:self.handle.close()
    def tokens(self,text):
        return request(self.url,'/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens']
    def detokenize(self,ids):
        return request(self.url,'/detokenize',{'tokens':ids,'render_special':True})['content']
    def completion(self,ids,allowance):
        return request(self.url,'/completion',{
            'prompt':ids,'n_predict':allowance,'temperature':0,'seed':POLICY['seed'],
            'repeat_penalty':POLICY['repetition_penalty'],'cache_prompt':False,
            'n_keep':-1,'return_tokens':True,'stream':False,'stop':[]},timeout=1800)

def load_tokenizer(root,key):
    from transformers import AutoTokenizer
    folder=Path(root)/'models/tokenizers'/key
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,fix_mistral_regex=False)
    spec=read_json(Path(root)/'manifests/models.json')[key]
    if digest(tok.chat_template)!=spec['chat_template_sha256']:
        raise ValueError('Frozen HF chat template changed')
    return tok

def _untreated_7b_runtime(key,representation,adapter):
    return adapter is None and representation=='tq1_0' and key in ('mistral7b','qwen7b')


def _prompt_length_bucket(input_tokens):
    if input_tokens<=2048:return 'le_2048'
    if input_tokens<=4096:return '2049_4096'
    if input_tokens<=8192:return '4097_8192'
    return 'gt_8192'


def infer(root,key,representation,adapter=None,limit=None,mirror=None):
    root=Path(root);verify_experiment(root)
    folder,artifact,manifest_path=artifact_paths(root,key,representation,adapter)
    if not manifest_path.exists():mirror.fetch(manifest_path.relative_to(root).as_posix(),manifest_path)
    if not manifest_path.exists():raise FileNotFoundError('Run artifact preparation first')
    saved=read_json(manifest_path)
    if not artifact.exists():mirror.fetch(artifact.relative_to(root).as_posix(),artifact)
    from streamlined.transport import file_hash
    if file_hash(artifact)!=saved['artifact_sha256']:raise ValueError('GGUF checksum mismatch')
    rows=selected_rows(root,representation,adapter)
    spec=read_json(root/'manifests/models.json')[key]
    tok=load_tokenizer(root,key)
    planned=[]
    for row in rows:
        prefix,_,_=render(tok,row['prompt_text'],'')
        ids=tok(prefix,add_special_tokens=False)['input_ids']
        planned.append((row,prefix,ids))
    native_context=int(spec['context'])
    context_reserve=backend_context_reserve(key,representation)
    needed=max((len(ids)+min(POLICY['max_new_tokens'],max(0,native_context-len(ids)))
                for _,_,ids in planned if len(ids)<native_context),default=1)
    context=min(native_context,needed)
    if limit is not None:planned=planned[:limit]
    fixed_7b_runtime=_untreated_7b_runtime(key,representation,adapter)
    run=stream(key,representation,adapter)
    rid_base=identity(root,key,representation,adapter,saved['artifact_sha256'],saved['calibration_sha256'])
    with Heartbeat(root,run,mirror,stall_seconds=1800) as hb:
        store=SyncStore(root,run,mirror)
        foreign={r['record_id'] for r in store.rows.values()
                 if r.get('config_hash')!=rid_base}
        if foreign:raise ValueError(f'{run}: existing records belong to another GGUF/configuration; use a new stream name')
        todo=[(row,prefix,ids) for row,prefix,ids in planned
              if not store.done(f'{rid_base}:{row["prompt_id"]}')]
        if fixed_7b_runtime:
            bucket_order={'le_2048':0,'2049_4096':1,'4097_8192':2,'gt_8192':3}
            # Scheduling only: the same persistent server, independent contexts,
            # prompts and decoding settings are used for every row.
            todo.sort(key=lambda item:(bucket_order[_prompt_length_bucket(len(item[2]))],
                                       len(item[2]),int(item[0]['prompt_id'])))
        if not todo:
            try:store.snapshot()
            finally:store.close()
            print(run,'already complete',flush=True);return
        options=({'slots':UNTREATED_TQ1_0_7B_SLOTS,'batch':2048,'ubatch':512,
                  'flash_attn':'default','threads':None,'verbosity':2}
                 if fixed_7b_runtime else None)
        parallel=min(options['slots'] if options else PARALLEL_SLOTS[key],len(todo))
        meter=Meter(root,run);started=time.perf_counter()
        summary={'generation_seconds':0.,'local_checkpoint_seconds':0.,'receiver_seconds':0.,
                 'csv_snapshot_seconds':0.,'input_tokens':0,'output_tokens':0,'new_rows':0}
        server_peak_rss=0;server=None
        status='failed'
        try:
            hb.update('server loading','load')
            server=Server(root,key,artifact,context,parallel,mirror,options=options)
            with server as backend:
                import psutil
                server_process=psutil.Process(backend.process.pid)
                # The frozen HF tokenizer defines the experimental input tokens.
                # Some GGUF pre-tokenizers segment the same Mistral text differently.
                # In that case llama-server still accepts the frozen IDs directly;
                # prove that its vocabulary maps them back to the exact prompt text.
                checks=planned[:5]+sorted(planned,key=lambda x:len(x[2]),reverse=True)[:2]
                tokenizer_checks={}
                for row,prefix,ids in checks:
                    got=backend.tokens(prefix)
                    match=got==ids
                    hf_roundtrip=backend.detokenize(ids)
                    gguf_roundtrip=backend.detokenize(got)
                    if hf_roundtrip!=prefix:
                        mismatch=next((i for i,(a,b) in enumerate(zip(hf_roundtrip,prefix)) if a!=b),
                                      min(len(hf_roundtrip),len(prefix)))
                        raise ValueError(f'GGUF vocabulary cannot reproduce frozen HF prompt at ID {row["prompt_id"]}; '
                                         f'first character mismatch {mismatch}, decoded={len(hf_roundtrip)}, expected={len(prefix)}')
                    first=next((i for i,(a,b) in enumerate(zip(got,ids)) if a!=b),
                               min(len(got),len(ids))) if not match else None
                    tokenizer_checks[str(row['prompt_id'])]=dict(hf_tokens=len(ids),gguf_tokens=len(got),
                        token_ids_equal=match,first_token_mismatch=first,
                        frozen_hf_ids_roundtrip_exact=True,
                        gguf_pretokenizer_roundtrip_exact=gguf_roundtrip==prefix,
                        gguf_pretokenizer_decoded_characters=len(gguf_roundtrip),
                        rendered_prompt_characters=len(prefix))
                tokenizer_ids_equal=all(x['token_ids_equal'] for x in tokenizer_checks.values())
                if not tokenizer_ids_equal:
                    print(f'{run}: GGUF pre-tokenizer differs from frozen HF tokenizer; '
                          'exact frozen HF token IDs passed to llama-server after lossless vocabulary round-trip check',flush=True)
                tokenizer_audit=root/'telemetry'/run/f'{hb.owner}_tokenizer_audit.json'
                write_json(tokenizer_audit,{
                    'run':run,'policy':'frozen_hf_token_ids_direct',
                    'rendered_prompt_text_unchanged':True,'all_checked_ids_equal':tokenizer_ids_equal,
                    'sampled_frozen_ids_roundtrip_exactly':True,
                    'per_row_roundtrip_required_before_generation':True,'checks':tokenizer_checks})
                try:mirror.put(root,tokenizer_audit)
                except Exception as error:print('Tokenizer audit upload pending:',error,flush=True)
                def generate(item):
                    row,prefix,ids=item;pid=row['prompt_id'];rid=f'{rid_base}:{pid}'
                    decoded=backend.detokenize(ids)
                    if decoded!=prefix:
                        mismatch=next((i for i,(a,b) in enumerate(zip(decoded,prefix)) if a!=b),
                                      min(len(decoded),len(prefix)))
                        raise ValueError(f'Frozen HF token IDs fail exact GGUF vocabulary round-trip at ID {pid}; '
                                         f'first character mismatch {mismatch}')
                    old=store.rows.get(rid,{})
                    base={k:row[k] for k in ('prompt_id','scenario','variant','step','split','group_id','ground_truth','prompt_sha256')}
                    base.update(record_id=rid,experiment_id=POLICY['experiment_id'],
                        quant_protocol='camlds_gguf_deployment_v1',model=key,
                        condition='untreated' if adapter is None else 'sft_'+adapter,
                        representation=representation,adapter=adapter or '',
                        prompt_text=row['prompt_text'],rendered_prompt_sha256=digest(prefix),
                        response_raw='',response_hash='',status='',
                        attempt_count=int(old.get('attempt_count',0))+1,
                        config_hash=rid_base,model_revision=saved['source_revision'],
                        gguf_sha256=saved['artifact_sha256'],gguf_bytes=saved['artifact_bytes'],
                        calibration_sha256=saved['calibration_sha256'],
                        imatrix_sha256=saved['imatrix_sha256'],
                        toolchain_commit=saved['toolchain_commit'],
                        backend='bitnet.cpp' if key=='bitnet2b' else 'llama.cpp',
                        input_tokens=len(ids),context=context,native_context=native_context,
                        backend_context_reserve_tokens=context_reserve,
                        backend_context_policy=('native_i2s_reserve_8_v1' if context_reserve else
                                                'nominal_context_v1'),
                        parallel_slots=parallel,server_total_context=context*parallel,
                        tokenizer_input_source='frozen_hf_token_ids_direct',
                        gguf_pretokenizer_ids_equal=tokenizer_ids_equal,
                        tokenizer_roundtrip_gate='exact_rendered_prompt_text_per_row',
                        max_new_tokens=max(0,min(POLICY['max_new_tokens'],
                            native_context-len(ids)-context_reserve)),
                        generation_settings={'temperature':0,'seed':POLICY['seed'],
                            'repeat_penalty':1.0,'cache_prompt':False,
                            'n_predict_max':POLICY['max_new_tokens'],
                            'backend_context_reserve_tokens':context_reserve},
                        output_allowance_reduced=(native_context-len(ids)-context_reserve)<POLICY['max_new_tokens'],
                        input_truncated=False,generation_error='')
                    if fixed_7b_runtime:
                        base.update(runtime_configuration_revision=UNTREATED_TQ1_0_7B_RUNTIME_REVISION,
                            prompt_scheduling='ascending_fixed_length_buckets_single_server',
                            prompt_length_bucket=_prompt_length_bucket(len(ids)),
                            server_batch_size=options['batch'],server_ubatch_size=options['ubatch'],
                            server_flash_attention=options['flash_attn'],
                            server_cpu_threads=options['threads'],
                            server_log_verbosity=options['verbosity'])
                    if not base['max_new_tokens']:
                        base.update(status='context_unsupported',finish_reason='context_unsupported',
                                    output_tokens=0,elapsed_seconds=0,**audit_response(''))
                    else:
                        begin=time.perf_counter()
                        try:
                            answer=backend.completion(ids,base['max_new_tokens'])
                            text=answer['content'];generated=answer.get('tokens',[])
                            if answer.get('truncated'):raise RuntimeError('Backend truncated the input or shifted context')
                            stop=answer.get('stop_type','unknown')
                            elapsed=time.perf_counter()-begin
                            base.update(status='completed',response_raw=text,response_hash=digest(text),
                                finish_reason=stop,response_truncated=stop=='limit',
                                output_tokens=len(generated),elapsed_seconds=elapsed,
                                output_tokens_per_second=len(generated)/elapsed if elapsed else None,
                                backend_timings=answer.get('timings',{}),
                                **audit_response(text))
                        except Exception as error:
                            base.update(status='retryable_error',generation_error=f'{type(error).__name__}: {error}')
                    return base
                hb.update(f'{len(todo)} rows','generating concurrently')
                failures=[]
                last_progress=time.monotonic()
                with ThreadPoolExecutor(max_workers=parallel,
                                        thread_name_prefix='independent-row') as executor:
                    futures={executor.submit(generate,item):item for item in todo}
                    for future in as_completed(futures):
                        hb.check()
                        base=future.result();pid=base['prompt_id']
                        timings=store.put(base)
                        summary['local_checkpoint_seconds']+=timings['local_seconds']
                        summary['receiver_seconds']+=timings['receiver_seconds']
                        summary['generation_seconds']+=float(base.get('elapsed_seconds',0))
                        summary['input_tokens']+=int(base.get('input_tokens',0))
                        summary['output_tokens']+=int(base.get('output_tokens',0))
                        summary['new_rows']+=1
                        if base['status']=='retryable_error':failures.append(pid)
                        try:server_peak_rss=max(server_peak_rss,server_process.memory_info().rss)
                        except psutil.Error:pass
                        hb.update(pid,'saved locally; receiver upload queued')
                        if (time.monotonic()-last_progress>=60) or summary['new_rows']==len(todo):
                            done=sum(store.done(f'{rid_base}:{item[0]["prompt_id"]}') for item in planned)
                            print(f'{run}: ID {pid} saved; {done}/{len(planned)}',flush=True)
                            last_progress=time.monotonic()
                if failures:
                    raise RuntimeError(f'{len(failures)} row generations failed and remain retryable')
            status='completed' if limit is None else 'pilot_completed'
        finally:
            if server is not None and server.log.exists():
                try:mirror.put(root,server.log)
                except Exception as error:print('Server log upload pending:',error,flush=True)
            try:
                begin=time.perf_counter()
                try:store.snapshot()
                finally:store.close()
                summary['csv_snapshot_seconds']+=time.perf_counter()-begin
                summary['receiver_seconds']=store.upload_seconds
            except Exception as error:
                print('Final CSV snapshot pending; row records remain authoritative:',error,flush=True)
            report=meter.close();report.update(summary)
            report.update(run=run,status=status,gguf_sha256=saved['artifact_sha256'],
                backend='bitnet.cpp' if key=='bitnet2b' else 'llama.cpp',
                backend_context_reserve_tokens=context_reserve,
                backend_context_policy=('native_i2s_reserve_8_v1' if context_reserve else
                                        'nominal_context_v1'),
                server_context=context,server_total_context=context*parallel,
                parallel_slots=parallel,request_contexts_independent=True,
                upload_batch_size=UPLOAD_BATCH_SIZE,upload_backpressure=UPLOAD_BACKPRESSURE,
                toolchain_commit=saved['toolchain_commit'],
                sampled_peak_server_rss_bytes=server_peak_rss,
                inference_and_io_wall_seconds=time.perf_counter()-started)
            if fixed_7b_runtime:
                report.update(runtime_configuration_revision=UNTREATED_TQ1_0_7B_RUNTIME_REVISION,
                    prompt_scheduling='ascending_fixed_length_buckets_single_server',
                    server_settings=options)
            target=root/'telemetry'/run/f'{hb.owner}.json';write_json(target,report)
            try:mirror.put(root,target)
            except Exception as error:print('Telemetry upload pending:',error,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','benchmark','infer'])
    parser.add_argument('--root',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--representation',required=True);parser.add_argument('--adapter')
    parser.add_argument('--pilot',type=int)
    args=parser.parse_args();root=Path(args.root)
    connection=read_json(root/'runtime/connection.json')
    mirror=Mirror(connection['url'],connection['token'])
    os.environ['HF_TOKEN']=mirror.call('/credential/hf')['token']
    with FileLock(str(root/'runtime'/f'{args.model}.quantized.lock'),timeout=0):
        if args.action=='prepare':
            run=stream(args.model,args.representation,args.adapter)+'/prepare'
            with Heartbeat(root,run,mirror,stall_seconds=3600) as hb:
                meter=Meter(root,run);status='failed';started=time.perf_counter()
                try:
                    artifact=prepare(root,args.model,args.representation,args.adapter,mirror)
                    print(artifact,flush=True);status='completed'
                finally:
                    report=meter.close();report.update(run=run,status=status,
                        preparation_and_upload_wall_seconds=time.perf_counter()-started)
                    path=root/'telemetry'/run/f'{hb.owner}.json';write_json(path,report)
                    try:mirror.put(root,path)
                    except Exception as error:print('Preparation telemetry upload pending:',error,flush=True)
        elif args.action=='benchmark':
            from .performance import benchmark
            run=stream(args.model,args.representation,args.adapter)+'/benchmark'
            with Heartbeat(root,run,mirror,stall_seconds=3600) as hb:
                hb.update('training-only performance pilot','benchmarking')
                benchmark(root,args.model,args.representation,mirror)
                hb.update('frozen performance profile saved','completed')
        else:infer(root,args.model,args.representation,args.adapter,args.pilot,mirror)

if __name__=='__main__':main()
