"""Frozen settings for the deployment-quantization continuation."""
from __future__ import annotations

from pathlib import Path
from streamlined.common import CONDITIONS, MODELS, POLICY, digest, read_json, verify_experiment

CONVENTIONAL = tuple(key for key in MODELS if key != 'bitnet2b')
LLAMA_CPP_COMMIT = 'a91eab67d8389d979d29fdf6633b2edd0718eb4b'
BITNET_CPP_COMMIT = '0b341e582afbf9e1011f24744b554c96a3477eb5'
BITNET_GGUF_REVISION = 'a1f2f1c765812aa8af3f6eda4a313707064bba15'
BITNET_GGUF_SHA256 = '4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162'
BITNET_GGUF_FILE = 'ggml-model-i2_s.gguf'
BITNET_GGUF_REPO = 'microsoft/bitnet-b1.58-2B-4T-gguf'
CALIBRATION_NAME = 'camlds_604_train_prompts_only'
SNAPSHOT_EVERY = 25
UPLOAD_BATCH_SIZE = 5
UPLOAD_BACKPRESSURE = 25
PARALLEL_SLOTS = {
    'qwen1b': 8,
    'hermes': 8,
    'bitnet2b': 4,
    'llama3b': 4,
    'mistral7b': 2,
    'qwen7b': 2,
}

# Untreated 7B TQ1_0 inference revision adopted after the 3 November runtime
# review.  It is deliberately scoped here rather than changing the default
# slot policy used by SFT and other representations.
UNTREATED_TQ1_0_7B_SLOTS = 4
UNTREATED_TQ1_0_7B_RUNTIME_REVISION = 'untreated_tq1_0_7b_four_slots_length_scheduled_v1'
NEW_JUDGE_RUN = 'quantized_common_cached_v1'
OPTIONAL_JUDGE_RUN = 'quantized_iq1_cached_v1'
JUDGE_PROTOCOL = 'uniform_common_v2_confidence_agreement'
QUANT_PROTOCOL = 'camlds_gguf_deployment_v1'

def condition_key(model: str, representation: str, adapter: str | None = None) -> str:
    if model not in MODELS:
        raise ValueError(model)
    if representation == 'native_i2_s':
        if model != 'bitnet2b': raise ValueError('Native I2_S is BitNet only')
    elif representation in ('tq1_0', 'iq1_s_camlds_calibrated'):
        if model == 'bitnet2b': raise ValueError('BitNet uses native I2_S only')
    else: raise ValueError(representation)
    if adapter is not None and adapter not in CONDITIONS: raise ValueError(adapter)
    return 'untreated' if adapter is None else 'sft_'+adapter

def stream(model: str, representation: str, adapter: str | None = None) -> str:
    stage = condition_key(model, representation, adapter)
    return f'quantized/{model}/{stage}/{representation}'

def selected_rows(root: Path, representation: str, adapter: str | None):
    from streamlined.data import records
    rows = records(root)
    if adapter is not None or representation == 'iq1_s_camlds_calibrated':
        rows = [row for row in rows if row['split'] in ('validation', 'test')]
    return rows

def calibration_rows(root: Path):
    from streamlined.data import records
    manifest = read_json(Path(root)/'manifests/experiment.json')
    eligible = {str(x) for x in manifest['eligible_ids']}
    rows = [row for row in records(root) if str(row['prompt_id']) in eligible]
    if len(rows) != 604 or any(row['split'] != 'train' for row in rows):
        raise ValueError('The frozen common training calibration set is not 604 train rows')
    return rows

def identity(root: Path, model: str, representation: str, adapter: str | None,
             artifact_sha256: str, calibration_sha256: str = '') -> str:
    frozen = verify_experiment(root)
    spec = read_json(Path(root)/'manifests/models.json')[model]
    return digest([QUANT_PROTOCOL, frozen['source'], spec['revision'], model,
                   condition_key(model, representation, adapter), representation,
                   artifact_sha256, calibration_sha256, POLICY['seed'],
                   POLICY['max_new_tokens'], POLICY['do_sample'],
                   POLICY['repetition_penalty'],
                   BITNET_CPP_COMMIT if model == 'bitnet2b' else LLAMA_CPP_COMMIT])
