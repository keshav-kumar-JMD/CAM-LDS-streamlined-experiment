from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from filelock import FileLock

MODELS = {
    'qwen1b': {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'size_b': 1.5, 'context': 32768},
    'hermes': {'repo': 'vicgalle/OpenHermes-Qwen1.5-1.8B', 'size_b': 1.8, 'context': 32768},
    'bitnet2b': {'repo': 'microsoft/bitnet-b1.58-2B-4T-bf16', 'size_b': 2, 'context': 4096},
    'llama3b': {'repo': 'meta-llama/Llama-3.2-3B-Instruct', 'size_b': 3, 'context': 131072},
    'mistral7b': {'repo': 'mistralai/Mistral-7B-Instruct-v0.3', 'size_b': 7, 'context': 32768},
    'qwen7b': {'repo': 'Qwen/Qwen2.5-7B-Instruct', 'size_b': 7, 'context': 32768},
}
CONDITIONS = {'technique': (1, 2), 'explanation': (1, 4),
              'joint': (1, 2, 4), 'full': (1, 2, 3, 4)}
CONFIDENCES = ['Certain: Attack', 'Almost Certain: Attack', 'Somewhat Certain: Attack',
               'Neutral', 'Somewhat Certain: Normal', 'Almost Certain: Normal', 'Certain: Normal']
POLICY = {'experiment_id': 'camlds_exact_teacher_v1', 'seed': 42, 'precision': 'bfloat16',
          'max_new_tokens': 1024, 'do_sample': False, 'repetition_penalty': 1.0,
          'epochs': 3, 'batch_size': 1, 'gradient_accumulation': 16, 'save_steps': 10,
          'training_context': 8192, 'test_conditions': list(CONDITIONS),
          'teacher_model': 'openai/gpt-5.5-20260423', 'teacher_first': 1981, 'teacher_last': 2970,
          'training_policy': 'universal-feasible-full-response-exact-distillation',
          'bitnet_policy':'BF16-master-with-native-online-ternary-weight-int8-activation-forward',
          'tokenizer_regex_policy':'checkpoint-original-no-mistral-regex-rewrite',
          'prompt_policy': 'source-text-unchanged', 'loss': 'equal-per-field-per-example',
          'haiku_model': 'claude-haiku-5-5', 'haiku_max_tokens': 128000,
          'judge_max_tokens': 8192, 'haiku_temperature': None,
          'haiku_thinking': {'type': 'disabled'}}

def digest(value):
    if isinstance(value, bytes): return hashlib.sha256(value).hexdigest()
    if not isinstance(value, str): value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def atomic(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = content if isinstance(content, bytes) else content.encode('utf-8')
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.partial')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        # Windows antivirus/Excel may briefly hold destination handles. Retry
        # replacement; never fall back to an in-place, partially visible write.
        for attempt in range(6):
            try:os.replace(temp,path);break
            except PermissionError:
                if attempt==5:raise
                time.sleep(.1*2**attempt)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def write_json(path, value): atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline=''); writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    atomic(path, out.getvalue())

def read_csv(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def upsert_csv(path, row, key='record_id'):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        rows = read_csv(path) if path.exists() else []
        indexed = {r[key]: r for r in rows}
        old = indexed.get(str(row[key]))
        if old and old.get('status') == 'completed' and old.get('response_hash') != row.get('response_hash'):
            raise ValueError('Refusing to overwrite a completed response with different content')
        indexed[str(row[key])] = row
        write_csv(path, indexed.values())

def secret(root, name):
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        p = parent / 'secret' / name
        if p.is_file():
            value = p.read_text(encoding='utf-8-sig').strip()
            if value: return value
    raise FileNotFoundError(f'secret/{name} was not found or is empty')

def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value): raise ValueError('Nonfinite JSON number')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)

def code_identity(root):
    return {p.name:digest(p.read_bytes()) for p in sorted((Path(root)/'streamlined').glob('*.py'))}

def verify_experiment(root):
    root=Path(root);manifest=read_json(root/'manifests/experiment.json')
    if manifest['policy']!=POLICY:raise ValueError('Frozen policy differs from running code')
    if manifest.get('code_sha256')!=code_identity(root):raise ValueError('Experiment code changed after freeze; complete preparation before launching new runs')
    for name,sha in manifest['input_csv_sha256'].items():
        if digest((root/'data'/name).read_bytes())!=sha:raise ValueError(f'Frozen input changed: {name}')
    if read_json(root/'manifests/models.json')!=manifest['models']:raise ValueError('Frozen model metadata changed')
    return manifest

def audit_response(text):
    import re
    result = dict(response_empty=not bool(text.strip()), valid_json=False,
                  schema_adherent=False, required_fields_present=False,
                  unexpected_text_present=False, parse_error='')
    try:
        obj = strict_json(text); result['valid_json'] = True
        required = {'techniques', 'confidence', 'explanation'}
        result['required_fields_present'] = isinstance(obj, dict) and required <= set(obj)
        if result['required_fields_present']:
            ts = obj['techniques']
            result['schema_adherent'] = (set(obj) == required and isinstance(ts, list)
                and len(ts) == 10 and all(isinstance(t, str) and re.fullmatch(r'T\d{4}(?:\.\d{3})?', t) for t in ts)
                and obj['confidence'] in CONFIDENCES and isinstance(obj['explanation'], str))
    except (ValueError, TypeError):
        result['parse_error'] = 'Whole response is not strict JSON'
        start, end = text.find('{'), text.rfind('}')
        if 0 <= start < end:
            try:
                strict_json(text[start:end+1])
                result['unexpected_text_present'] = bool(text[:start].strip() or text[end+1:].strip())
            except (ValueError, TypeError): pass
    return result
