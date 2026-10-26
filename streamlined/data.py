"""Extract source text without cleaning, reordering, repairing or relabelling."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
from collections import Counter
from .common import *

def object_spans(text):
    """Top-level JSON value character spans, retaining the exact lexical text."""
    decoder = json.JSONDecoder(); pos = 0
    def ws(i):
        while i < len(text) and text[i].isspace(): i += 1
        return i
    pos = ws(pos)
    if text[pos:pos+1] != '{': return {}
    pos += 1; spans = {}
    while True:
        pos = ws(pos)
        if text[pos:pos+1] == '}': return spans
        key, pos = decoder.raw_decode(text, pos); pos = ws(pos)
        if text[pos] != ':': raise ValueError('Missing colon')
        start = ws(pos+1); _, end = decoder.raw_decode(text, start)
        if key in spans: raise ValueError('Duplicate source key')
        spans[key] = (start, end); pos = ws(end)
        if text[pos:pos+1] == '}': return spans
        if text[pos:pos+1] != ',': raise ValueError('Missing comma')
        pos += 1

def prepare(root, source, old_split, queries):
    root, source = Path(root), Path(source)
    raw = source.read_bytes(); source_hash = digest(raw)
    marker = root/'manifests/source.json'
    if marker.exists() and read_json(marker)['source_sha256'] != source_hash:
        raise ValueError('Source changed: create a new experiment instead of overwriting this one')
    lines = raw.decode('utf-8-sig').splitlines()
    qrows = [json.loads(s) for s in Path(queries).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    groups = read_json(old_split)['groups']
    mapping = {tuple(g['group_id']): g['partition'] for g in groups}
    prompts, teachers, splits, truths, audits = [], [], [], [], []
    for pid, lineno in enumerate(range(POLICY['teacher_first'], POLICY['teacher_last']+1), 1):
        line = lines[lineno-1]; obj = json.loads(line); ref = qrows[pid-1]
        if obj['canonical_slug'] != POLICY['teacher_model']: raise ValueError('Wrong teacher row')
        for key in ('scenario', 'variant', 'step'):
            if obj[key] != ref[key]: raise ValueError(f'Identity mismatch at {pid}: {key}')
        if obj['ground_truth'] != ref['ground_truth']: raise ValueError('Ground truth mapping mismatch')
        prompt = obj['prompt']
        if not isinstance(prompt, str): raise ValueError('Source prompt is not text')
        a, b = object_spans(line)['response']; lexical = line[a:b]
        # Structured stored responses have no original API whitespace beyond this JSON.
        # String-valued responses are decoded once to the exact stored string.
        answer = obj['response'] if isinstance(obj['response'], str) else lexical
        identity = dict(prompt_id=pid, scenario=obj['scenario'], variant=obj['variant'], step=obj['step'])
        split = mapping[(str(obj['scenario']), str(obj['step']))]
        prompts.append(dict(identity, prompt_text=prompt, prompt_sha256=digest(prompt), source_row=lineno))
        teachers.append(dict(identity, teacher_model=obj['canonical_slug'], teacher_response_raw=answer,
                             teacher_response_sha256=digest(answer), source_value_text=lexical,
                             source_representation=type(obj['response']).__name__, split=split))
        splits.append(dict(identity, group_id=json.dumps([obj['scenario'], obj['step']]), split=split, split_seed=42))
        truths.append(dict(identity, ground_truth=obj['ground_truth']))
        hints = re.findall(r'mitre_(?:technique|tactic)_(?:id|name)|["\']ground[_ ]truth["\']|\bkey=', prompt, re.I)
        audits.append(dict(identity, annotation_warning=bool(hints), warning_matches=hints,
                           answer_ids_in_prompt=[x for x in obj['ground_truth'] if x in prompt],
                           teacher_prompt_unmodified=True, **audit_response(answer)))
    if len(prompts) != 990 or len({p['prompt_id'] for p in prompts}) != 990:
        raise ValueError('Expected 990 uniquely mapped prompt IDs')
    identities=Counter((p['scenario'],p['variant'],p['step']) for p in prompts)
    for audit in audits:
        audit['metadata_identity_occurrences']=identities[(audit['scenario'],audit['variant'],audit['step'])]
    for name, rows in [('prompts',prompts), ('teacher_responses',teachers), ('split_manifest',splits),
                       ('ground_truth',truths), ('dataset_audit',audits)]:
        write_csv(root/'data'/f'{name}.csv',rows)
    for split in ('train','validation','test'):
        write_csv(root/'data'/f'teacher_responses_{split}.csv',[r for r in teachers if r['split']==split])
    dest = root/'data/source/llm_responses.json'; dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve(): shutil.copyfile(source,dest)
    manifest = dict(source_sha256=source_hash, teacher=POLICY['teacher_model'], first_row=1981,
                    last_row=2970, rows=990, split_counts=dict(Counter(r['split'] for r in splits)),
                    original_split_sha256=digest(Path(old_split).read_bytes()),
                    source_representation_note='Object responses preserve original source JSON substring; API-original serialization unavailable.',
                    prompts_sha256=digest(prompts), teachers_sha256=digest(teachers))
    write_json(marker,manifest); write_json(root/'configs/experiment.json',POLICY)
    return manifest

def records(root):
    root = Path(root)
    prompts = read_csv(root/'data/prompts.csv')
    joins = [{r['prompt_id']:r for r in read_csv(root/'data'/name)}
             for name in ('teacher_responses.csv','split_manifest.csv','ground_truth.csv')]
    for row in prompts:
        for index in joins: row.update(index[row['prompt_id']])
        if digest(row['prompt_text']) != row['prompt_sha256']: raise ValueError('Prompt changed')
        if digest(row['teacher_response_raw']) != row['teacher_response_sha256']: raise ValueError('Teacher changed')
    return prompts

def freeze_tokenizers(root):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoConfig
    root = Path(root); token = secret(root,'HF_Token.txt'); os.environ['HF_TOKEN'] = token
    lockpath = root/'manifests/models.json'
    with FileLock(str(lockpath)+'.lock'):
        lock = read_json(lockpath) if lockpath.exists() else {}
        for key, profile in MODELS.items():
            if key not in lock:
                lock[key] = dict(profile, revision=HfApi(token=token).model_info(profile['repo']).sha)
                write_json(lockpath,lock)
            spec = lock[key]; folder = root/'models/tokenizers'/key
            cached=spec.get('tokenizer_files_sha256',{})
            intact=bool(cached) and all((folder/name).exists() and digest((folder/name).read_bytes())==sha
                                       for name,sha in cached.items())
            if not intact:
                snapshot_download(spec['repo'], revision=spec['revision'], token=token,
                                  local_dir=folder, allow_patterns=['*.json','*.model','*.txt','*.jinja'])
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True,fix_mistral_regex=False)
            config = AutoConfig.from_pretrained(folder, local_files_only=True)
            if not tok.is_fast or not tok.chat_template: raise ValueError(f'{key}: fast chat tokenizer required')
            native = int(getattr(config,'max_position_embeddings',profile['context']))
            spec['context'] = min(native,profile['context'])
            spec['training_context'] = min(spec['context'], POLICY['training_context'])
            spec['chat_template_sha256'] = digest(tok.chat_template)
            spec['tokenizer_files_sha256']={p.name:digest(p.read_bytes()) for p in folder.iterdir()
                if p.is_file() and p.suffix in ('.json','.model','.txt','.jinja')}
            write_json(lockpath,lock)
    return lock

def feasibility(root):
    from transformers import AutoTokenizer
    from .encoding import encode_exact, render
    root=Path(root); rows=records(root); specs=read_json(root/'manifests/models.json')
    audit=[]; excluded=set()
    for key,spec in specs.items():
        tok=AutoTokenizer.from_pretrained(root/'models/tokenizers'/key,local_files_only=True,fix_mistral_regex=False)
        for row in rows:
            prefix,full,_=render(tok,row['prompt_text'],row['teacher_response_raw'])
            input_n=len(tok(prefix,add_special_tokens=False)['input_ids'])
            required=len(tok(full,add_special_tokens=False)['input_ids'])
            fits=required<=spec['training_context']
            enc=encode_exact(tok,row)
            mask_missing=[c for c,fields in CONDITIONS.items() if not any(x in fields and x!=1 for x in enc['fields'])]
            if row['split']=='train' and not fits: excluded.add(row['prompt_id'])
            audit.append(dict(prompt_id=row['prompt_id'],model=key,split=row['split'],
                prompt_tokens=input_n,teacher_and_ending_tokens=required-input_n,
                required_tokens=required,context=spec['context'],training_context=spec['training_context'],
                excess_tokens=max(0,required-spec['training_context']),fits_training=fits,
                generation_allowance=max(0,min(POLICY['max_new_tokens'],spec['context']-input_n)),
                conditional_mask_unavailable=mask_missing))
    write_csv(root/'data/context_feasibility.csv',audit)
    exclusions=[dict(r,reason='complete teacher example exceeds at least one model training context',
                     universal_exclusion=True) for r in audit if r['split']=='train' and r['prompt_id'] in excluded]
    write_csv(root/'data/training_exclusions.csv',exclusions, list(audit[0])+['reason','universal_exclusion'])
    ids=[r['prompt_id'] for r in rows if r['split']=='train' and r['prompt_id'] not in excluded]
    manifest=dict(eligible_ids=ids, excluded_ids=sorted(excluded,key=int), source=read_json(root/'manifests/source.json'),
                  models=specs,policy=POLICY,code_sha256=code_identity(root),
                  input_csv_sha256={name:digest((root/'data'/name).read_bytes()) for name in
                     ('prompts.csv','teacher_responses.csv','split_manifest.csv','ground_truth.csv')})
    manifest['config_hash']=digest(manifest)
    target=root/'manifests/experiment.json'
    if target.exists() and read_json(target)['config_hash']!=manifest['config_hash']:
        if any((root/'results').rglob('records/*.json')) or any((root/'models/adapters').glob('*.zip')):
            raise ValueError('Cannot refreeze after experiments have started; create a new experiment folder')
    write_json(target,manifest)
    return {'eligible_train':len(ids),'excluded_train':len(excluded),'rows_audited':len(audit)}

