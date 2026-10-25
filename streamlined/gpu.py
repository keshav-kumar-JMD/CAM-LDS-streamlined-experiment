"""BF16 HF baseline and exact-text LoRA loss ablations; no quantized fallback."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
from filelock import FileLock
from .common import *
from .data import records
from .encoding import encode_exact,render
from .monitor import Heartbeat
from .storage import Store,persist
from .telemetry import Meter
from .transport import Mirror,package,safe_extract,file_hash

def load_base(root,key,mirror,hb,training=False):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('Connect this notebook to a Colab GPU runtime')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen protocol requires native BF16. Use A100, H100 or L4; no silent precision change.')
    spec=read_json(root/'manifests/models.json')[key]
    folder=root/'models/base'/key; archive=root/'models/base'/f'{key}.zip'
    started=time.perf_counter()
    marker=folder/'snapshot_complete.json'
    complete=read_json(marker) if marker.exists() else {}
    intact=complete.get('revision')==spec['revision'] and all((folder/name).is_file() and
        (folder/name).stat().st_size==size for name,size in complete.get('files',{}).items())
    if not intact:
        hb.update('restore/download base model','model cache')
        restored=mirror.fetch(f'models/base/{key}.zip',archive) if mirror else False
        if restored:safe_extract(archive,folder)
        else:
            snapshot_download(spec['repo'],revision=spec['revision'],local_dir=folder,
                token=os.environ.get('HF_TOKEN'),allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
            write_json(marker,{'revision':spec['revision'],'files':{
                p.relative_to(folder).as_posix():p.stat().st_size for p in folder.rglob('*')
                if p.is_file() and '.cache' not in p.parts and p.name!='snapshot_complete.json'}})
            package(folder,archive)
    if not marker.exists() or read_json(marker)['revision']!=spec['revision']:
        raise ValueError('Restored base snapshot does not match the frozen revision')
    if not archive.exists():package(folder,archive)
    persist(root,archive,mirror) # Reconcile an upload interrupted after download completed.
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,fix_mistral_regex=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if digest(tok.chat_template)!=spec['chat_template_sha256']:raise ValueError('Tokenizer template changed')
    cache_seconds=time.perf_counter()-started;started=time.perf_counter()
    hb.update('load BF16 weights','loading')
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    quant=getattr(model.config,'quantization_config',None)
    if hasattr(quant,'to_dict'):quant=quant.to_dict()
    if key=='bitnet2b':
        if not quant or quant.get('quantization_mode')!='online' or quant.get('linear_class')!='autobitlinear':
            raise ValueError('BitNet requires its native trainable BF16-master online quantization path')
    elif quant:raise ValueError('Deployment quantization is outside this experiment')
    model.config.use_cache=not training
    return tok,model,spec,{'model_load_seconds':time.perf_counter()-started,'model_cache_restore_download_sync_seconds':cache_seconds}

def run_identity(root,key,condition):
    m=verify_experiment(root)
    return digest([m['config_hash'],key,condition,POLICY])

def infer(root,key,condition='untreated_hf',split='all',mirror=None):
    import torch
    from transformers import set_seed
    from peft import PeftModel
    root=Path(root); run=f'{key}/{condition}/{split}';stream=run
    identity=run_identity(root,key,condition);rows=[r for r in records(root) if split=='all' or r['split']==split]
    with Heartbeat(root,run,mirror,stall_seconds=1800) as hb:
        store=Store(root,stream,mirror)
        # Replay locally spooled rows before generating new answers.
        if mirror:
            for r in list(store.rows.values()):mirror.call('/response',{'stream':stream,'row':r})
        todo=[r for r in rows if not store.done(f'{identity}:{r["prompt_id"]}')]
        if not todo:print(f'{run}: already complete',flush=True);return
        meter=Meter(root,run);status='failed';session_input=0;session_output=0;inference_seconds=0
        try:
            tok,model,spec,load_seconds=load_base(root,key,mirror,hb)
            adapter_hash=''
            if condition!='untreated_hf':
                artifact=root/'models/adapters'/f'{key}_{condition}.zip'
                if mirror:mirror.fetch(f'models/adapters/{key}_{condition}.zip',artifact)
                if not artifact.exists():raise FileNotFoundError('Train and durably save this adapter first')
                folder=root/'models/adapters'/key/condition;safe_extract(artifact,folder)
                meta=read_json(folder/'identity.json')
                if meta['identity']!=identity:raise ValueError('Adapter protocol mismatch')
                adapter_hash=file_hash(artifact);model=PeftModel.from_pretrained(model,folder,local_files_only=True)
            model.eval();set_seed(POLICY['seed']);generation_start=time.perf_counter()
            for row in todo:
                hb.check();pid=row['prompt_id'];rid=f'{identity}:{pid}'
                existing=store.rows.get(rid,{})
                base={k:row[k] for k in ('prompt_id','scenario','variant','step','split','group_id','ground_truth','prompt_sha256')}
                base.update(record_id=rid,experiment_id=POLICY['experiment_id'],model=key,condition=condition,
                    prompt_text=row['prompt_text'],response_raw='',response_hash='',status='in_progress',
                    attempt_count=int(existing.get('attempt_count',0))+1,config_hash=identity,
                    model_revision=spec['revision'],adapter_sha256=adapter_hash,precision='bfloat16',
                    generation_settings=POLICY,generation_error='',finish_reason='',input_truncated=False,
                    gpu=torch.cuda.get_device_name(),bitnet_intrinsic_quantization=key=='bitnet2b')
                store.put(base);hb.update(pid,'generating')
                prefix,_,_=render(tok,row['prompt_text'],'')
                inputs=tok(prefix,add_special_tokens=False,return_tensors='pt').to('cuda')
                n=inputs['input_ids'].shape[-1];allowance=max(0,min(POLICY['max_new_tokens'],spec['context']-n))
                base.update(input_tokens=n,max_new_tokens=allowance,output_allowance_reduced=allowance<POLICY['max_new_tokens'])
                if not allowance:
                    base.update(status='context_unsupported',finish_reason='context_unsupported',output_tokens=0,
                                elapsed_seconds=0,**audit_response(''))
                    store.put(base);hb.update(pid,'context unsupported recorded');continue
                start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=allowance,do_sample=False,num_beams=1,
                            repetition_penalty=1.0,pad_token_id=tok.pad_token_id)
                    torch.cuda.synchronize();ids=output[0,n:];text=tok.decode(ids,skip_special_tokens=True)
                    eos=model.generation_config.eos_token_id or tok.eos_token_id
                    eos=[eos] if isinstance(eos,int) else eos
                    reason='eos' if len(ids) and int(ids[-1]) in (eos or []) else 'length' if len(ids)>=allowance else 'other'
                    seconds=time.perf_counter()-start
                    session_input+=n;session_output+=len(ids);inference_seconds+=seconds
                    base.update(status='completed',response_raw=text,response_hash=digest(text),
                        finish_reason=reason,response_truncated=reason=='length',output_tokens=len(ids),
                        elapsed_seconds=seconds,output_tokens_per_second=len(ids)/seconds,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        **audit_response(text))
                except Exception as error:
                    base.update(status='retryable_error',generation_error=f'{type(error).__name__}: {error}')
                    store.put(base);raise
                store.put(base);hb.update(pid,'saved locally')
                print(f'{run}: ID {pid} saved ({sum(r.get("status")=="completed" for r in store.rows.values())}/{len(rows)})',flush=True)
            status='completed'
        finally:
            report=meter.close();report.update(run=run,status=status,config_hash=identity)
            if 'load_seconds' in locals():report.update(load_seconds)
            report.update(session_input_tokens=session_input,session_output_tokens=session_output,inference_seconds=inference_seconds)
            path=root/'telemetry'/run/f'{hb.owner}.json';write_json(path,report);persist(root,path,mirror)
            if 'model' in locals():del model
            gc.collect();torch.cuda.empty_cache()

def train(root,key,condition,mirror=None):
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    root=Path(root);run=f'{key}/train/{condition}';identity=run_identity(root,key,condition)
    path=root/'models/adapters'/f'{key}_{condition}.zip'
    with Heartbeat(root,run,mirror,stall_seconds=3600) as hb:
        if mirror:mirror.fetch(f'models/adapters/{key}_{condition}.zip',path)
        if path.exists():
            folder=root/'models/adapters'/key/condition;safe_extract(path,folder)
            if read_json(folder/'identity.json')['identity']!=identity:raise ValueError('Existing adapter identity differs')
            print(f'{key}/{condition} adapter already complete',flush=True);return
        meter=Meter(root,run);status='failed'
        checkpoint_root=root/'checkpoints'/key/condition;checkpoint_root.mkdir(parents=True,exist_ok=True)
        pointer=checkpoint_root/'latest.json'
        if mirror:mirror.fetch(f'checkpoints/{key}/{condition}/latest.json',pointer)
        resume=None
        if pointer.exists():
            ptr=read_json(pointer)
            if ptr['identity']!=identity:raise ValueError('Checkpoint belongs to another configuration')
            archive=root/ptr['archive']
            if mirror:mirror.fetch(ptr['archive'],archive)
            if file_hash(archive)!=ptr['sha256']:raise ValueError('Training checkpoint checksum differs')
            resume=checkpoint_root/f'checkpoint-{ptr["step"]}';safe_extract(archive,resume)
        try:
            tok,model,spec,load_seconds=load_base(root,key,mirror,hb,training=True)
            ids=set(read_json(root/'manifests/experiment.json')['eligible_ids'])
            train_rows=[r for r in records(root) if r['prompt_id'] in ids]
            encoded=[];unavailable=[]
            for row in train_rows:
                item=encode_exact(tok,row)
                if len(item['input_ids'])>spec['training_context']:raise ValueError('Frozen context feasibility changed')
                if condition!='full' and not any(f in CONDITIONS[condition] and f!=1 for f in item['fields']):
                    unavailable.append({'prompt_id':row['prompt_id'],'condition':condition,
                        'reason':'No safely identifiable target field; exact response retained in data'})
                else:encoded.append(item)
            audit=root/'telemetry'/key/f'{condition}_mask_exclusions.csv'
            write_csv(audit,unavailable,['prompt_id','condition','reason']);persist(root,audit,mirror)
            if not encoded:raise ValueError('No training examples for this condition')
            set_seed(POLICY['seed']);large=key in ('qwen7b','mistral7b');rank=8 if large else 16
            model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=.05,
                target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
            model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            def collate(items):
                length=max(len(x['input_ids']) for x in items)
                return {'input_ids':torch.tensor([x['input_ids']+[tok.pad_token_id]*(length-len(x['input_ids'])) for x in items]),
                        'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(length-len(x['input_ids'])) for x in items]),
                        'fields':torch.tensor([x['fields']+[0]*(length-len(x['fields'])) for x in items]),
                        'labels':torch.tensor([x['labels']+[-100]*(length-len(x['labels'])) for x in items])}
            class FieldTrainer(Trainer):
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    fields=inputs.pop('fields')[:,1:];labels=inputs.pop('labels')[:,1:]
                    out=model(**inputs);logits=out.logits[:,:-1,:]
                    losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),
                        labels.reshape(-1),reduction='none',ignore_index=-100).reshape(labels.shape)
                    values=[]
                    for index in range(len(labels)):
                        parts=[]
                        for f in CONDITIONS[condition]:
                            mask=(fields[index]==f)&(labels[index]!=-100)
                            if mask.any():parts.append(losses[index][mask].mean())
                        if not parts:
                            if condition=='full':parts=[losses[index][labels[index]!=-100].mean()]
                            else:raise ValueError('No enabled loss tokens')
                        values.append(torch.stack(parts).mean())
                    loss=torch.stack(values).mean()
                    return (loss,out) if return_outputs else loss
            class Save(TrainerCallback):
                def on_step_end(self,args,state,control,**kwargs):
                    hb.check();hb.update(state.global_step,'training')
                def on_save(self,args,state,control,**kwargs):
                    folder=checkpoint_root/f'checkpoint-{state.global_step}'
                    runtime=meter.snapshot();write_json(folder/'runtime.json',runtime)
                    archive=checkpoint_root/f'step-{state.global_step}.zip';package(folder,archive)
                    persist(root,archive,mirror)
                    write_json(pointer,dict(identity=identity,step=state.global_step,
                        archive=archive.relative_to(root).as_posix(),sha256=file_hash(archive)))
                    persist(root,pointer,mirror);hb.update(state.global_step,'checkpoint saved locally')
            runtime=meter.snapshot()
            if resume:
                previous=read_json(resume/'runtime.json')
                if previous['packages']!=runtime['packages']:
                    raise RuntimeError('Training software versions changed; restore the recorded versions before resume')
            args=TrainingArguments(output_dir=str(checkpoint_root),num_train_epochs=POLICY['epochs'],
                per_device_train_batch_size=1,gradient_accumulation_steps=16,learning_rate=1e-4 if large or key=='bitnet2b' else 2e-4,
                bf16=True,fp16=False,save_strategy='steps',save_steps=POLICY['save_steps'],save_total_limit=2,
                logging_steps=1,report_to='none',remove_unused_columns=False,seed=42,data_seed=42,
                warmup_ratio=.05,weight_decay=.01,gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant':False},dataloader_num_workers=0)
            trainer=FieldTrainer(model=model,args=args,train_dataset=encoded,data_collator=collate,callbacks=[Save()])
            trainer.model_accepts_loss_kwargs=False
            hb.update('trainer started','training');training_start=time.perf_counter()
            trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            training_seconds=time.perf_counter()-training_start
            final=root/'models/adapters'/key/condition;trainer.save_model(final);tok.save_pretrained(final)
            write_json(final/'identity.json',dict(identity=identity,model=key,condition=condition,
                trained_ids=[x['prompt_id'] for x in encoded],policy=POLICY,model_spec=spec,
                objective='field-masked exact-text distillation; masked fields remain visible context'))
            package(final,path);persist(root,path,mirror);status='completed';hb.update('adapter saved locally')
        finally:
            report=meter.close();report.update(run=run,status=status,config_hash=identity)
            if 'load_seconds' in locals():report.update(load_seconds)
            if 'training_start' in locals():report['training_wall_seconds_including_checkpoint_transfers']=time.perf_counter()-training_start
            out=root/'telemetry'/run/f'{hb.owner}.json';write_json(out,report);persist(root,out,mirror)
            if 'trainer' in locals():del trainer
            if 'model' in locals():del model
            gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['infer','train']);p.add_argument('--root',required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--condition',default='untreated_hf')
    p.add_argument('--split',default='all');a=p.parse_args();root=Path(a.root)
    connection=read_json(root/'runtime/connection.json');mirror=Mirror(connection['url'],connection['token'])
    os.environ['HF_TOKEN']=mirror.call('/credential/hf')['token']
    with FileLock(str(root/'runtime'/f'{a.model}.worker.lock'),timeout=0):
        if a.action=='train':train(root,a.model,a.condition,mirror)
        else:infer(root,a.model,a.condition,a.split,mirror)
if __name__=='__main__':main()
