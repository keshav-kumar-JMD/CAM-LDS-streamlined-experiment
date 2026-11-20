"""Generate the 04--08 continuation notebooks without modifying frozen 00--03."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_notebooks import BOOTSTRAP, LOCAL_SETUP, cell, notebook, save
from streamlined.common import CONDITIONS
from continuation.protocol import CONVENTIONAL

SETUP_COMMANDS = r'''**One-time laptop setup, in separate PowerShell terminals:**

1. Stop the prior `streamlined.receiver` terminal if it is still serving port 8770. Let any active 01--03 worker finish first. Start the extended receiver:

```powershell
Set-Location 'D:\reit4841_code\1-58-bit-llm-CAM-LDS\streamlined_experiment'
.\.venv-local\Scripts\python.exe -m continuation.receiver --root . --port 8770
```

2. Start the HTTPS tunnel in another terminal and keep it open:

```powershell
cloudflared tunnel --url http://127.0.0.1:8770
```

3. Start the independent audible watchdog in a third terminal:

```powershell
Set-Location 'D:\reit4841_code\1-58-bit-llm-CAM-LDS\streamlined_experiment'
.\.venv-local\Scripts\python.exe -m streamlined.monitor --root .
```

4. Read the local receiver token and test sound in a fourth terminal:

```powershell
Get-Content .\runtime\receiver_token.txt
.\.venv-local\Scripts\python.exe -m streamlined.monitor --root . --test
```

Use the same URL/token for distinct model notebooks. Never launch the same model and condition twice. Keep Windows awake, audio unmuted, tunnel, receiver and watchdog terminals open. A lost Colab runtime stops its heartbeat; the local watchdog alarms without relying on the notebook kernel.
'''

def setup():
    nb=notebook('# 04 — Quantized continuation setup\n\nRun locally before any 04--06 GPU notebook. '
        'Select `streamlined_experiment/.venv-local/Scripts/python.exe` as the kernel. '
        'The receiver extends the original service, retains its token and port, and adds atomic row checkpoints plus 25-row CSV snapshots.',True)
    cell(nb,'1. Locate and verify the frozen experiment','Read the completed 00--03 manifest and confirm that the new source code can use it.',
        LOCAL_SETUP+'''\nfrom streamlined.common import verify_experiment
from continuation.protocol import calibration_rows
print('Experiment:',verify_experiment(ROOT)['config_hash'])
print('Frozen calibration rows:',len(calibration_rows(ROOT)))''',
        'After a local kernel restart, rerun 1. Source and old results are read only.')
    cell(nb,'2. Receiver, HTTPS tunnel and audible watchdog',
        'Run the commands below in external laptop terminals. The original receiver must be stopped before launching the extended receiver on the same port. '
        '**Existing active workers should finish before replacing the receiver.**',
        '''from streamlined.transport import Mirror
from streamlined.common import read_json
from pathlib import Path
import urllib.request, json
token = (ROOT/'runtime/receiver_token.txt').read_text().strip()
print('Token file exists:',bool(token))
print('Extended receiver must be running in its own terminal before any Colab bootstrap.')
print('Watchdog state:',read_json(ROOT/'runtime/watchdog_alive.json') if (ROOT/'runtime/watchdog_alive.json').exists() else 'not running')
del token''',
        'If the tunnel or receiver stops, restart the commands and rerun 2. No model rows are repeated.',SETUP_COMMANDS)
    cell(nb,'3. Input and expected result counts',
        'Condition A: native BitNet I2_S. Condition B: TQ1_0 for five conventional models. '
        'Condition C: optional IQ1_S with only the frozen 604 training prompts as importance-matrix data. '
        'Create the exact 604-prompt calibration corpus locally for audit and Colab reuse. '
        'Untreated A/B infer 990 rows; every SFT adapter and optional C run infers 140 validation plus 145 test rows.',
        '''from continuation.judge import expected_streams
from continuation.artifacts import calibration_text
corpus,sha = calibration_text(ROOT)
print('Local calibration corpus:',corpus,'SHA256:',sha)
for family in ('required','optional'):
    streams = expected_streams(family)
    print(family, 'candidate streams:',len(streams), 'untreated:',sum(x[2] is None for x in streams),
          'SFT:',sum(x[2] is not None for x in streams))''',
        'Rerun 1 then 3 at any time. No GPU or API work occurs.')
    save(nb,'04_quantized_setup.ipynb')

def gpu(model,representation,sft,number):
    condition='SFT adapters' if sft else 'untreated base'
    label=('native I2_S' if representation=='native_i2_s' else
           'TQ1_0' if representation=='tq1_0' else 'IQ1_S with CAM-LDS training calibration')
    optional=representation=='iq1_s_camlds_calibrated'
    pre_change_counts={'qwen1b':239,'hermes':990,'llama3b':658,
                       'mistral7b':252,'qwen7b':0,'bitnet2b':0}
    process_history=(
        '\n\n**Inference process revision (2 November 2026).** Earlier saved 04 responses used one '
        'llama-server slot and synchronously acknowledged each row through the laptop receiver. '
        'The revised process uses model-specific concurrent slots for independent one-row requests, '
        'atomic Colab records, background receiver uploads in batches of up to five, backpressure at '
        '25 outstanding rows, and laptop CSV snapshots every 25 mutations. GGUF construction, '
        'rendered prompts, frozen HF token IDs, per-row context allowance, and generation settings '
        'are unchanged. The old and new processes can coexist within the same resumable stream; '
        'the per-row parallel_slots field distinguishes new records. '
        + (f'Before the revision, this 04 run had approximately {pre_change_counts[model]}/990 '
           f'response rows saved ({pre_change_counts[model]/990:.0%}); this count comes from '
           'results/quantized record files, with telemetry used to check completed runs. '
           'It is a dated snapshot and may be lower than current coverage.'
           if not sft and not optional else
           'Before the revision, this 05 notebook had 0/285 response rows saved for each of its '
           'four adapter streams, according to results/quantized record files. This refers to '
           'inference results, not preparation of adapter GGUF artifacts.'
           if sft and not optional else
           'This optional 06 condition was not part of the pre-revision 04/05 progress tally.')
    )
    nb=notebook(f'# {number:02d} — {model} {condition}, {label}\n\n'
        'Select a **Colab GPU** kernel in VS Code. Read `04_quantized_setup.ipynb` for the laptop receiver, tunnel, and audible watchdog. '
        'Use only the pinned frozen base/adapters. Inference sends the same complete, unmodified teacher-row prompt with the frozen chat template, '
        'greedy decoding, seed 42, repetition penalty 1 and 1,024-token maximum. '
        'Every row is an independent llama-server completion with prompt caching disabled; prompt identity and tokenization are checked. '
        'An empty or malformed model response is still a completed response. Context-unsupported rows retain an explicit status. '
        + ('This optional condition uses the 604 frozen training prompts for the GGUF importance matrix; validation/test prompts never enter calibration.'
           if optional else 'This condition does not use CAM-LDS importance-matrix calibration.')
        + process_history)
    cell(nb,'1. Restore code and data from laptop',
         'Start the extended receiver, tunnel and watchdog using notebook 04 setup. Paste the HTTPS URL and receiver token. '
         'The receiver sends frozen code, manifests, prompts and tokenizer files; it does not send API/HF secrets in the bundle.',
         BOOTSTRAP,
         'New runtime or tunnel: rerun 1, then 2, 3, and the interrupted stage. The same URL/token may be reused across different models.')
    cell(nb,'2. Install pinned worker requirements',
         '**Installations and built toolchains are reused within the same Colab runtime. A new runtime reinstalls Python packages but restores the locally saved, pinned CUDA build only when its GPU architecture and CUDA toolkit match.** '
         'The frozen base model is downloaded directly from Hugging Face and cached within that runtime. Final quantized artifacts are saved on the laptop.',
         '''import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-colab.txt')],check=True)
print('Worker dependencies ready')''',
         'After runtime reset rerun 1 then 2; ordinary cell reruns reuse installed packages.')
    cell(nb,'3. Verify receiver, watchdog, configuration and GPU',
         'The same authenticated receiver serves all distinct model notebooks. Check GPU identity and available memory before construction.',
         f'''from streamlined.common import read_json, verify_experiment, CONDITIONS
from streamlined.transport import Mirror
from continuation.notebook import worker
from continuation.protocol import condition_key
import subprocess
MODEL={model!r}
REPRESENTATION={representation!r}
connection=read_json(ROOT/'runtime/connection.json')
mirror=Mirror(connection['url'],connection['token'])
health=mirror.call('/health')
if not health.get('watchdog_alive'):raise RuntimeError('Start the laptop audible watchdog')
print(health, verify_experiment(ROOT)['config_hash'])
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                               '--format=csv,noheader'],text=True))''',
         'After connection errors rerun 1 and 3. After a new runtime rerun 1–3. Resolve duplicate live worker leases before retrying.')
    cell(nb,'4. Prepare the pinned GGUF artifact'+('s' if sft else ''),
         ('Merge each complete HF adapter into its frozen BF16 base, convert to GGUF, then construct '+label+'. '
          if sft else 'Construct '+label+' from the frozen untreated base. ')+
         ('The 604 exact eligible training prompts form the importance-matrix corpus; matrix hash and source IDs are saved. '
          if optional else '')+
         'The laptop receives each final GGUF artifact and its checksum before the notebook advances. '
         '**This stage can spend substantial GPU/CPU time and local disk; completed artifacts and toolchains are reused on rerun within a live runtime.** '
         'BitNet SFT export uses a compatibility gate and stops with a clear error if this pinned native path cannot convert the merged checkpoint.',
         ('''for adapter in CONDITIONS:
    worker(ROOT,MODEL,REPRESENTATION,adapter,action='prepare')''' if sft else
          '''worker(ROOT,MODEL,REPRESENTATION,action='prepare')'''),
         'Rerun 1–3 after a runtime change, then 4. A fully uploaded final artifact is reused; failed conversions leave no completed artifact manifest.',
         cost='GPU/CPU construction and potentially multi-GB upload; no paid Haiku calls.')
    cell(nb,'5. Five-row inference pilot',
         'Run five frozen IDs per stream through the final server and save them as real results. The full run skips these IDs. '
         'The GGUF tokenizer is compared to the frozen HF tokenizer before generation. Frozen HF token IDs define the experimental input. '
         'If the GGUF pre-tokenizer segments identical text differently, the backend must detokenize the frozen IDs back to the exact rendered prompt before generation; the difference is retained in a tokenizer audit. '
         'The pilot uses the same per-request context and model-specific parallel-slot policy as the full run. Each row remains a separate request and context. '
         'Inspect the server log for actual CUDA offload and token/s before the long run.',
         ('''for adapter in CONDITIONS:
    worker(ROOT,MODEL,REPRESENTATION,adapter,action='infer',pilot=5)''' if sft else
          '''worker(ROOT,MODEL,REPRESENTATION,action='infer',pilot=5)'''),
         'Rerun 1–3 after a new runtime, then 5. Saved pilot IDs are not repeated.',
         cost='GPU inference for up to five rows per stream.')
    cell(nb,'6. Complete independent inference',
         ('All four adapters are run on the frozen validation and test IDs only. '
          if sft else 'The untreated run covers all 990 frozen prompt IDs. ' if not optional else
          'The optional IQ1_S untreated run covers validation and test IDs only. ')+
         'Independent rows are submitted concurrently using a conservative model-specific slot count; completion order may differ, but every result is keyed to its frozen prompt ID. '
         'Each response is first written atomically as a JSON record on Colab. A background uploader sends laptop checkpoints in batches of five, applies backpressure at 25 queued rows, and fully flushes at completion. '
         'The laptop CSV snapshot is refreshed every 25 saved rows and at normal completion. '
         'The same prompt IDs are skipped on resume; errors are retryable. GPU, latency, token and I/O metrics are saved as telemetry.',
         ('''for adapter in CONDITIONS:
    worker(ROOT,MODEL,REPRESENTATION,adapter,action='infer')''' if sft else
          '''worker(ROOT,MODEL,REPRESENTATION,action='infer')'''),
         'After interruption rerun 1–3 and 6. Rerun 4 only if the GGUF artifact was never completed; section 5 is optional on resume. '
         'The per-row JSON records, including rows after the latest CSV snapshot, are authoritative.',
         cost='GPU inference for remaining rows; no paid Haiku calls.')
    cell(nb,'7. Verify coverage and snapshot',
         'Read current laptop-backed record counts. Check that each required stream is terminal before launching the Haiku judge.',
         '''from continuation.store import SyncStore
from continuation.protocol import selected_rows, stream
from collections import Counter
for adapter in (CONDITIONS if 'CONDITIONS' in globals() else __import__('streamlined.common',fromlist=['CONDITIONS']).CONDITIONS) if '''+str(sft)+''' else (None,):
    name=stream(MODEL,REPRESENTATION,adapter)
    with SyncStore(ROOT,name,mirror) as store:
        store.snapshot()
        expected=len(selected_rows(ROOT,REPRESENTATION,adapter))
        states=Counter(row['status'] for row in store.rows.values())
        print(name,dict(states), 'expected:',expected,'saved:',len(store.rows))''',
         'Rerun 1–3 then 7 after a runtime change. Missing IDs are handled by rerunning 6.')
    if not sft and representation=='tq1_0' and model in ('mistral7b','qwen7b'):
        document_7b_fixed_runtime(nb,model)
    if sft and representation=='tq1_0' and model in ('llama3b','mistral7b','qwen7b'):
        document_sft_l4_runtime(nb,model)
    save(nb,f'{number:02d}_{"sft" if sft else "untreated"}_{representation}_{model}.ipynb')

def document_7b_fixed_runtime(nb,model):
    """Document the fixed four-slot continuation without adding a tuning stage."""
    transition_counts={'mistral7b':360,'qwen7b':257}
    intro=''.join(nb['cells'][0]['source'])
    intro+=(f'\n\n**Fixed 7B runtime revision (3 November 2026).** At this transition, the laptop '
            f'held {transition_counts[model]}/990 completed {model} rows. Remaining rows use one persistent '
            'llama-server with four independent slots, batch 2,048, microbatch 512, backend-default Flash '
            'Attention and backend-default CPU threads. Missing prompts are queued in fixed token-length '
            'groups (up to 2,048; 4,096; 8,192; and over 8,192 tokens) without restarting the server. '
            'The earlier mandatory performance sweep and architecture-specific profile are retired because '
            'their cost was disproportionate and no faster profile was established. Existing rows remain '
            'untouched and are skipped on resume. New rows record the runtime revision, slot count, server '
            'settings and scheduling bucket. Prompt text, frozen HF token IDs, GGUF weights, sampling, '
            'independent contexts, natural EOS and the original 1,024-token ceiling remain unchanged. '
            'There is no identical-prefix cache.')
    nb['cells'][0]['source']=intro.splitlines(True)
    for item in nb['cells']:
        if item.get('cell_type')!='markdown':continue
        body=''.join(item['source'])
        if body.startswith('## 5.'):
            body=body.replace('model-specific parallel-slot policy','fixed four-slot policy')
            body+='\n\nThis is an experimental five-row response pilot, not a runtime tuning sweep. It uses the same fixed configuration and persistent-server implementation as Section 6.'
        elif body.startswith('## 6.'):
            body=body.replace('a conservative model-specific slot count','four independent slots')
            body+='\n\nFor this untreated 7B TQ1_0 run, missing rows are ordered by frozen prompt-token length and sent through one persistent server. Length groups change queue order only; they do not change prompts, contexts, generation settings or the output ceiling.'
        item['source']=body.splitlines(True)


def document_sft_l4_runtime(nb,model):
    """Document the output-neutral L4 revision for unfinished 05 notebooks."""
    slots={'llama3b':1,'mistral7b':2,'qwen7b':2}[model]
    intro=''.join(nb['cells'][0]['source'])
    intro+=(f'\n\n**L4 runtime revision (4 November 2026).** This revision applies only to the '
            f'unfinished {model} 05 notebook; Hermes was already complete and Qwen1B was already '
            f'running, so neither is changed. At adoption, all four {model} adapter streams held '
            '0/285 response rows. The pinned llama.cpp toolchain is compiled in Release mode for '
            'the allocated GPU architecture (L4 compute capability 8.9 / `sm_89`) and a matching '
            'saved toolchain is restored when available. Completed GGUF artifacts are restored from '
            'the laptop instead of being rebuilt. Each adapter uses one persistent server with '
            f'{slots} independent slots, batch 2,048, microbatch 512, backend-default Flash Attention '
            'and backend-default CPU threads. Missing prompts are ordered by frozen prompt-token '
            'length using fixed buckets without changing prompt text, token IDs, contexts or decoding. '
            f'The five-row pilot retains diagnostic verbosity {3 if model=="llama3b" else 2}; the full run uses verbosity 1 while '
            'retaining per-row records, errors and final telemetry. Run only one inference notebook '
            'per L4. Atomic local records, five-row background receiver batches and 25-row '
            'backpressure remain enabled. Sampling, natural EOS and the 1,024-token ceiling are unchanged.')
    if model=='llama3b':
        intro+=('\n\n**Llama 3B single-slot correction (5 November 2026).** The four initial five-row '
            'adapter pilots measured only 5.94–6.05 aggregate output tokens/s, 3.1–3.2% mean '
            'sampled GPU utilization and 5.62–5.63 GiB peak GPU memory on an L4; all 20 responses '
            'reached the unchanged 1,024-token ceiling. A prior single-slot Llama 3B TQ1_0 server '
            'log measured 20.4–20.7 output tokens/s. Subsequent missing rows therefore use one '
            'persistent slot and one per-row context. Batch 2,048, microbatch 512, backend-default '
            'Flash Attention, backend-default CPU threads, prompts, token IDs and decoding remain '
            'unchanged. Pilot verbosity 3 retains explicit CUDA layer-offload evidence; full-run '
            'verbosity remains 1. Existing pilot rows are preserved with their original metadata.')
    nb['cells'][0]['source']=intro.splitlines(True)
    for item in nb['cells']:
        if item.get('cell_type')!='markdown':continue
        body=''.join(item['source'])
        if body.startswith('## 4.'):
            body+='\n\nOn a new L4 runtime, a matching saved `sm_89` Release toolchain and completed GGUF artifacts are restored when available. Quantization is not repeated when the final artifact manifest and checksum are present.'
        elif body.startswith('## 5.'):
            body+=f'\n\nThe pilot retains server verbosity {3 if model=="llama3b" else 2} for CUDA-offload and throughput diagnostics. It uses the same slots, batch, microbatch and decoding settings as Section 6.'
        elif body.startswith('## 6.'):
            body+='\n\nUse one notebook per L4. Missing rows are scheduled from shorter to longer frozen prompt-token buckets through one persistent server. The full run lowers only server log verbosity to 1; response generation and research telemetry are unchanged.'
        item['source']=body.splitlines(True)

def judge():
    nb=notebook('# 07 — Haiku judge for quantized deployment conditions and reference streams\n\n'
        'Select the **local** `.venv-local` kernel. The same frozen six-dimension common rubric from notebook 03 judges '
        'one quantized candidate per independent API request, with the corresponding Haiku reference and ground truth. '
        'The static rubric alone is marked for prompt caching; candidate cases are never cached or combined. '
        'Cache hits are opportunistic in batches and actual cache usage is retained per response. '
        'Condition C, the GPT-5.5 teacher and the trivial baselines use separate ledgers. The teacher and baselines cover the same 285 validation+test IDs.',True)
    cell(nb,'1. Local environment and frozen source',
         'Read source, code and interpreter identity. No tunnel or local receiver is needed for local Haiku API calls.',
         LOCAL_SETUP+'''\nfrom streamlined.common import verify_experiment
print('Frozen:',verify_experiment(ROOT)['config_hash'])''',
         'After local kernel restart rerun 1, then the interrupted section.')
    cell(nb,'2. Confirm API access and planned request scope',
         'Load `secret/Claude_API_Key.txt` from the project tree. The key is held in memory and never printed. '
         'Before paying for new requests, all required candidate streams must be complete or explicitly context-unsupported.',
         '''from streamlined.api import client
from continuation.judge import source_rows
api=client(ROOT)
print('Model:',api.models.retrieve('claude-haiku-5-5').id)
required_rows,coverage=source_rows(ROOT,'required')
print('Paid judge requests planned:',len(required_rows))
print('Required streams:',len(coverage))
print('Expected 12,449 if BitNet retains its previous context-unsupported pattern; actual count is sourced from new records.')''',
         'If an input stream is incomplete, finish its 04/05 section 6 first, then rerun 1–2.',
         cost='Model metadata lookup only; no judge requests.')
    cell(nb,'3. Freeze common cached judge jobs',
         'Construct identical one-row judge cases with the notebook 03 rubric. '
         'Only the static system rubric gets a cache-control marker. '
         'The request manifest freezes model, rubric, prompt and candidate identity; reruns reuse the same paid ledger.',
         '''from continuation.judge import make_run
required=make_run(ROOT,'required')
print(required.counts())
print('Cache policy:',required.folder/'cache_policy.json')''',
         'Rerun 1–3 after a kernel restart. If source candidates change, finish their own run and use a new judge protocol name; this ledger refuses changes.',
         cost='No paid judge requests.')
    cell(nb,'4. Paid five-row pilot',
         'Inspect five real judgments, raw output and token/cache usage. All pilot rows are checkpointed and skipped by the main batch.',
         '''print(required.pilot(size=5,retry_failed=True))
print(required.export())''',
         'Rerun 1–3 then 4 only if the pilot is incomplete. Unknown outcomes remain held; inspect receipts and resolve them before an explicit retry.',
         cost='Up to five paid Messages API requests.')
    cell(nb,'5. Reconcile ambiguous batch submissions',
         'Find a previously submitted remote batch by exact custom-ID set; this step never creates a new paid request.',
         '''print(required.reconcile())
print(required.export())''',
         'Rerun 1–3 then 5 after a network interruption. Repeat until unknown submissions are resolved.',
         cost='Read-only API listing and result retrieval.')
    cell(nb,'6. Submit remaining required candidates',
         'Batch one candidate per API request. Submitted IDs are held in the local ledger, so rerunning does not duplicate them. '
         'Rate-limit rejections can be retried after waiting; unknown outcomes must be reconciled first.',
         '''print(required.submit(batch_size=50,retry_failed=True))
print(required.export())''',
         'Rerun 1–3, 5, then 6. Do not erase the submissions ledger; a 429 may need a cooldown.',
         cost='Paid Message Batches API for remaining required candidates; prompt-cache hits are not guaranteed.')
    cell(nb,'7. Poll and collect required results',
         'Collect ended batches. Every returned raw judgment and usage object is saved as an atomic local JSON record. '
         'The CSV is refreshed every 100 record mutations and at the end of this cell; each record is saved immediately.',
         '''print(required.poll(max_seconds=3600))
print(required.export())''',
         'Rerun 1–3 then 7 until no pending batch remains. Polling never submits new paid work.',
         cost='Read-only batch polling and result collection.')
    cell(nb,'8. Optional IQ1_S: freeze jobs',
         'Run after all five optional IQ1_S untreated and 20 adapter streams finish. '
         'This separate ledger preserves the same rubric/cache design and never alters required judgments.',
         '''optional=make_run(ROOT,'optional')
print(optional.counts())''',
         'Rerun 1 and 8 after a kernel restart. If IQ1_S is not being performed, skip sections 8–11.',
         cost='No paid judge requests.')
    cell(nb,'9. Optional IQ1_S: five-row pilot',
         'Verify one-row judgments and actual cache usage before optional batch submission.',
         '''print(optional.pilot(size=5,retry_failed=True))
print(optional.export())''',
         'Rerun 1 and 8–9 as needed; completed pilot rows are skipped.',
         cost='Up to five paid Messages API requests.')
    cell(nb,'10. Optional IQ1_S: reconcile and submit',
         'Resolve unknown batch outcomes before submitting optional judgments.',
         '''print(optional.reconcile())
print(optional.submit(batch_size=50,retry_failed=True))
print(optional.export())''',
         'Rerun 1, 8, then 10. Unknown submissions remain held to avoid duplicate charges.',
         cost='Paid Message Batches API for remaining optional candidates.')
    cell(nb,'11. Optional IQ1_S: poll results',
         'Collect, checkpoint and export optional judgments.',
         '''print(optional.poll(max_seconds=3600))
print(optional.export())''',
         'Rerun 1, 8, then 11 until pending reaches zero; polling does not submit.',
         cost='Read-only batch polling and result collection.')
    cell(nb,'12. GPT-5.5 teacher: verify 285 rows and freeze jobs',
         'Use the exact stored GPT-5.5 teacher responses as anonymous candidates for all 140 validation and 145 test prompts. '
         'The unchanged common rubric, ground truth and Haiku reference are used in a separate frozen ledger.',
         '''from continuation.judge import source_rows, make_run
teacher_rows,teacher_coverage=source_rows(ROOT,'teacher')
print('Teacher validation+test candidates:',len(teacher_rows))
print('Coverage:',teacher_coverage)
teacher=make_run(ROOT,'teacher')
print(teacher.counts())
print('Frozen teacher ledger:',teacher.folder)''',
         'After a kernel restart rerun Sections 1 and 12. Once frozen, the same 285 request identities are reloaded.',
         cost='Source verification only; no paid judge requests.')
    cell(nb,'13. GPT-5.5 teacher: paid five-row pilot',
         'Judge five teacher rows before batch submission; completed pilot IDs are skipped later.',
         '''print(teacher.pilot(size=5,retry_failed=True))
print(teacher.export())''',
         'Rerun Sections 1, 12 and 13 only if the pilot remains incomplete.',
         cost='Up to five paid Messages API requests.')
    cell(nb,'14. GPT-5.5 teacher: reconcile and submit',
         'Reconcile any ambiguous prior submission, then submit only teacher jobs not completed or pending.',
         '''print(teacher.reconcile())
print(teacher.submit(batch_size=50,retry_failed=True))
print(teacher.export())''',
         'Rerun Sections 1 and 12, then 14. Never delete the submissions ledger.',
         cost='Paid Message Batches requests for remaining teacher rows.')
    cell(nb,'15. GPT-5.5 teacher: poll, checkpoint and export',
         'Collect teacher judgments as atomic local records and refresh the CSV snapshot.',
         '''print(teacher.poll(max_seconds=3600))
print(teacher.export())''',
         'Rerun Sections 1 and 12, then 15 until pending is zero. Run Notebook 08 afterward.',
         cost='Read-only batch polling and collection.')
    cell(nb,'16. Reference baselines: verify 570 rows and freeze jobs',
         'Create null and training-only modal-technique candidates for each of the 285 evaluation prompts. '
         'Both pass through the identical common judge protocol and a separate frozen ledger.',
         '''import importlib
import continuation.protocol as continuation_protocol
import continuation.judge as continuation_judge
importlib.reload(continuation_protocol)
continuation_judge=importlib.reload(continuation_judge)
source_rows,make_run=continuation_judge.source_rows,continuation_judge.make_run
baseline_rows,baseline_coverage=source_rows(ROOT,'baselines')
print('Baseline candidates:',len(baseline_rows))
print('Coverage:',baseline_coverage)
baselines=make_run(ROOT,'baselines')
print(baselines.counts())''',
         'After a kernel restart rerun Sections 1 and 16.',cost='No paid requests.')
    cell(nb,'17. Reference baselines: paid five-row pilot',
         'Judge five baseline cases; empty completed responses remain eligible.',
         '''print(baselines.pilot(size=5,retry_failed=True))
print(baselines.export())''','Rerun Sections 1, 16 and 17 if incomplete.',cost='Up to five paid requests.')
    cell(nb,'18. Reference baselines: reconcile and submit',
         'Reconcile ambiguous submissions, then submit remaining baseline jobs.',
         '''print(baselines.reconcile())
print(baselines.submit(batch_size=50,retry_failed=True))
print(baselines.export())''','Rerun Sections 1 and 16, then 18.',cost='Paid batch requests.')
    cell(nb,'19. Reference baselines: poll, checkpoint and export',
         'Collect atomic baseline judgment records and refresh the CSV.',
         '''print(baselines.poll(max_seconds=3600))
print(baselines.export())''','Rerun Sections 1 and 16, then 19 until pending is zero.',cost='Read-only polling.')
    save(nb,'07_haiku_judge_quantized.ipynb')

def final():
    nb=notebook('# 08 — Consolidate final experiment results\n\n'
        'Select the **local** `.venv-local` kernel. Build one clearly named results folder from 00–03 and new 04–07 runs. '
        'Every candidate stream with at least one authoritative row record is exported, including declared partial streams. '
        'Coverage and incompleteness remain explicit in the manifest; no missing rows are silently filled.',True)
    cell(nb,'1. Local environment and lineage check',
         'Verify frozen prompts, model metadata and source hashes.',
         LOCAL_SETUP+'''\nfrom streamlined.common import verify_experiment
print('Frozen:',verify_experiment(ROOT)['config_hash'])''',
         'Rerun 1 after a local kernel restart.')
    cell(nb,'2. Export all available response streams and judge snapshots',
         'Join validation and test rows for every SFT adapter and optional IQ1_S response. '
         'Keep untreated A/B runs as 990-row files, IQ1_S untreated as 285-row files, and each judge run separately. '
         'Partial streams are exported with their available records. A manifest reports expected, completed, '
         'context-unsupported, missing and other status counts plus file hashes.',
         '''from continuation.consolidate import consolidate
print(consolidate(ROOT,include_optional=True))
print('Manifest:',ROOT/'results/final/08__manifest__files.csv')''',
         'Rerun 1–2 after any additional completion. Files are rebuilt atomically from authoritative row records; no paid or GPU work occurs.')
    save(nb,'08_consolidate_results.ipynb')

def compute_summary():
    nb=notebook('# 09 — Per-stream compute, retry and telemetry accounting\n\n'
        'Select the **local** `.venv-local` kernel. Aggregate every structured telemetry invocation and response-record '
        'attempt without combining distinct model/condition/representation streams. Preparation, training and inference '
        'remain separate phases. Pilot work is separate, Colab billing is excluded, and unavailable historical transfer '
        'timings are reported as unavailable rather than assigned to inference.',True)
    cell(nb,'1. Local environment and lineage check',
         'Verify the frozen experiment before reading telemetry or response records.',
         LOCAL_SETUP+'''\nfrom streamlined.common import verify_experiment
print('Frozen:',verify_experiment(ROOT)['config_hash'])''',
         'Rerun section 1 after a local kernel restart.')
    cell(nb,'2. Refresh response exports and build auditable compute summaries',
         'First refresh all available response exports, including declared partial streams. Then create the invocation '
         'inventory, per-row performance audit, exact-stream resource summaries, retry/coverage '
         'counts, model-level resource totals, API token/cache usage and separate per-stream CSVs. Retries and resumes '
         'are summed only within the exact stream. Compute files are organized under `results/final/compute/` in '
         '`invocations`, `rows`, and `stream_summary` subfolders. Every filename is prefixed with its source notebook '
         '(`01`--`07`); cross-notebook summaries use `09`. Model totals are resource totals, not elapsed time.',
         '''from continuation.consolidate import consolidate
from continuation.compute import summarize
print('Response export:',consolidate(ROOT,include_optional=True))
report=summarize(ROOT)
print(report)
print('Results manifest:',ROOT/'results/final/compute/09__manifest__compute_files.csv')''',
         'Rerun sections 1–2 after new or recovered runs. Outputs rebuild locally; no GPU or paid API work occurs.')
    cell(nb,'3. Inspect principal report tables',
         'Display the stream accounting and coverage/retry tables from their categorized compute subfolders.',
         '''import csv

def preview(folder, name, columns, limit=10):
    path=ROOT/'results/final/compute'/folder/name
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    print(f'\\n{name}: {len(rows)} rows')
    print(' | '.join(columns))
    for row in rows[:limit]:print(' | '.join(str(row.get(column,'')) for column in columns))

preview('stream_summary','09__compute__stream_summary.csv',['model','condition','representation','phase','invocation_count','experimental_resource_gpu_hours','completed','total_retries'])
preview('rows','09__compute__coverage_and_retries.csv',['model','condition','representation','record_count','completed','context_unsupported','retryable_error','total_retries'])
preview('stream_summary','09__compute__model_resource_summary.csv',['model','representation','phase','stream_count','invocation_count','resource_gpu_hours','estimated_energy_wh'])''',
         'Rerun section 3 any time after section 2; it is read-only.')
    save(nb,'09_compute_telemetry_summary.ipynb')

def main():
    setup()
    for model in CONVENTIONAL:
        gpu(model,'tq1_0',False,4)
        gpu(model,'tq1_0',True,5)
        gpu(model,'iq1_s_camlds_calibrated',False,6)
        gpu(model,'iq1_s_camlds_calibrated',True,6)
    gpu('bitnet2b','native_i2_s',False,4)
    gpu('bitnet2b','native_i2_s',True,5)
    judge();final();compute_summary()

if __name__=='__main__':main()
