"""Build pinned GGUF toolchains and checkpoint-specific deployment artifacts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter
from streamlined.common import POLICY, digest, read_json, secret, verify_experiment, write_json
from streamlined.transport import file_hash, package, safe_extract
from .protocol import (BITNET_CPP_COMMIT, BITNET_GGUF_FILE, BITNET_GGUF_REPO,
    BITNET_GGUF_REVISION, BITNET_GGUF_SHA256, CALIBRATION_NAME,
    LLAMA_CPP_COMMIT, calibration_rows, condition_key)

BITNET_CONVERTER_PATCH='multi_shard_filename_tuple_v1'

def run(command, root, log_name, env=None):
    log=Path(root)/'telemetry/logs'/log_name
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as output:
        output.write('COMMAND '+json.dumps([str(x) for x in command])+'\n');output.flush()
        result=subprocess.run([str(x) for x in command],stdout=output,stderr=subprocess.STDOUT,
                              cwd=root,env=env,check=False)
    if result.returncode:
        tail=log.read_text(encoding='utf-8',errors='replace')[-4000:]
        raise RuntimeError(f'{log_name} exited {result.returncode}; see {log}\n{tail}')
    return log

def cmake_commands(repo, build, arch, bitnet=False, jobs=None):
    """Return configure/build commands for the selected pinned backend."""
    configure=['cmake','-S',repo,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
               '-DGGML_CUDA=ON',f'-DCMAKE_CUDA_ARCHITECTURES={arch}']
    if bitnet:
        # The pinned BitNet setup explicitly enables these llama.cpp targets
        # and builds the default target set rather than a named subtarget.
        configure.extend(['-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_EXAMPLES=ON',
                          '-DLLAMA_BUILD_COMMON=ON','-DLLAMA_BUILD_SERVER=ON'])
    build_command=['cmake','--build',build,'--config','Release','-j',
                   str(jobs or min(8,max(2,(os.cpu_count() or 4)//2)))]
    if bitnet:
        # Once the pinned BitNet project has explicitly enabled its server,
        # build only that required target.  Its default all target also builds
        # unrelated examples/tools, any of which can fail after libllama has
        # already compiled and needlessly block native-I2_S inference.
        build_command.extend(['--target','llama-server'])
    else:
        build_command.extend(['--target','llama-server','llama-quantize','llama-imatrix'])
    return configure,build_command

def patch_bitnet_converter(repo):
    """Make the pinned converter's multi-shard filename lists reusable.

    BitnetModel performs two tensor passes.  At the pinned upstream commit,
    multi-shard names are generator expressions, so the scale scan consumes
    them and the conversion pass writes a zero-tensor GGUF.  Tuples preserve
    the identical ordered filenames for both passes.
    """
    path=Path(repo)/'utils/convert-hf-to-gguf-bitnet.py'
    text=path.read_text(encoding='utf-8')
    expressions=(
        'return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))',
        'return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))',
    )
    changed=False
    for expression in expressions:
        replacement=expression.replace('return (','return tuple(')
        if expression in text:
            text=text.replace(expression,replacement,1);changed=True
        elif replacement not in text:
            raise RuntimeError('Pinned BitNet converter layout changed; compatibility patch cannot be verified')
    if changed:path.write_text(text,encoding='utf-8')
    return BITNET_CONVERTER_PATCH

def toolchain(root, bitnet=False, mirror=None):
    root=Path(root)
    name='bitnet_cpp' if bitnet else 'llama_cpp'
    commit=BITNET_CPP_COMMIT if bitnet else LLAMA_CPP_COMMIT
    repo=root/'.cache/toolchains'/name
    if not (repo/'.git').is_dir():
        repo.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','--filter=blob:none','--no-checkout',
             'https://github.com/microsoft/BitNet.git' if bitnet else 'https://github.com/ggml-org/llama.cpp.git',repo],
            root,name+'_clone.log')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=commit or not (repo/'CMakeLists.txt').is_file():
        run(['git','-C',repo,'checkout','--detach',commit],root,name+'_checkout.log')
    if bitnet:run(['git','-C',repo,'submodule','update','--init','--recursive'],root,name+'_submodules.log')
    converter_patch=patch_bitnet_converter(repo) if bitnet else ''
    build=repo/'build'
    server=build/'bin/llama-server'
    quantizer=build/'bin/llama-quantize'
    imatrix=build/'bin/llama-imatrix'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Use the Colab GPU runtime for GGUF construction')
    major,minor=torch.cuda.get_device_capability()
    arch=f'{major}{minor}'
    cuda_version=(torch.version.cuda or 'unknown').replace('.','_')
    build_marker=build/'camlds_build.json'
    previous=read_json(build_marker) if build_marker.exists() else {}
    marker={'commit':commit,'cuda_arch':arch,'cuda_version':cuda_version,
            'build_profile':'bitnet_server_target_v3' if bitnet else 'llama_selected_targets_v1'}
    archive=root/'models/toolchains'/name/commit/f'cuda_{arch}_toolkit_{cuda_version}.zip'
    if (mirror and previous!=marker):
        if not archive.exists():mirror.fetch(archive.relative_to(root).as_posix(),archive)
        if archive.exists():
            safe_extract(archive,build/'bin')
            for path in (build/'bin').iterdir():
                if path.is_file():path.chmod(path.stat().st_mode|0o111)
            required=[server] if bitnet else [server,quantizer,imatrix]
            if all(path.exists() for path in required):
                probe=subprocess.run([str(server),'--help'],stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,timeout=30,check=False)
                if probe.returncode==0:
                    write_json(build_marker,marker)
                    previous=read_json(build_marker)
                    print(f'Reused laptop CUDA toolchain for architecture {arch}',flush=True)
    if (not server.exists() or (not bitnet and (not quantizer.exists() or not imatrix.exists()))
            or previous!=marker):
        configure,build_command=cmake_commands(repo,build,arch,bitnet=bitnet)
        run(configure,root,name+'_configure.log')
        run(build_command,root,name+'_build.log')
        if not server.exists():
            produced=sorted(p.name for p in (build/'bin').glob('*')) if (build/'bin').is_dir() else []
            raise RuntimeError(f'{name} build completed without llama-server; build/bin contains {produced}')
        write_json(build_marker,marker)
        if mirror:
            package(build/'bin',archive)
            mirror.put(root,archive)
    if bitnet:
        package_marker=repo/'build/camlds_gguf_python.json'
        if not package_marker.exists():
            run([sys.executable,'-m','pip','install','--quiet',str(repo/'3rdparty/llama.cpp/gguf-py')],
                root,'bitnet_gguf_python.log')
            write_json(package_marker,{'commit':commit})
    return dict(repo=repo,server=server,quantizer=quantizer,imatrix=imatrix,commit=commit,
                converter_patch=converter_patch)

def base_snapshot(root, key):
    from huggingface_hub import snapshot_download
    root=Path(root);spec=read_json(root/'manifests/models.json')[key]
    folder=root/'models/base'/key
    marker=folder/'snapshot_complete.json'
    complete=read_json(marker) if marker.exists() else {}
    intact=(complete.get('revision')==spec['revision'] and
            all((folder/name).is_file() and (folder/name).stat().st_size==size
                for name,size in complete.get('files',{}).items()))
    if not intact:
        snapshot_download(spec['repo'],revision=spec['revision'],local_dir=folder,
            token=os.environ.get('HF_TOKEN'),allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
        write_json(marker,{'revision':spec['revision'],'files':{
            p.relative_to(folder).as_posix():p.stat().st_size for p in folder.rglob('*')
            if p.is_file() and '.cache' not in p.parts and p.name!='snapshot_complete.json'}})
    return folder

def merged_snapshot(root,key,adapter,mirror):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root=Path(root);base=base_snapshot(root,key)
    if adapter is None:return base,''
    archive=root/'models/adapters'/f'{key}_{adapter}.zip'
    if not archive.exists():mirror.fetch(f'models/adapters/{key}_{adapter}.zip',archive)
    if not archive.exists():raise FileNotFoundError(f'Missing final {key}/{adapter} adapter archive')
    adapter_hash=file_hash(archive)
    destination=root/'.cache/merged'/key/adapter
    marker=destination/'merged_identity.json'
    if marker.exists() and read_json(marker).get('adapter_sha256')==adapter_hash:
        if list(destination.glob('*.safetensors')):return destination,adapter_hash
    extracted=root/'.cache/extracted_adapters'/key/adapter
    safe_extract(archive,extracted)
    source_spec=read_json(root/'manifests/models.json')[key]
    if read_json(extracted/'identity.json').get('model_spec',source_spec)!=source_spec:
        raise ValueError('Adapter source specification differs from frozen model')
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,extracted,local_files_only=True)
    try:model=model.merge_and_unload(safe_merge=True)
    except Exception as error:
        raise RuntimeError(f'{key}/{adapter} cannot be safely merged; native export gate failed: {error}') from error
    destination.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(destination,safe_serialization=True,max_shard_size='4GB')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,fix_mistral_regex=False)
    tokenizer.save_pretrained(destination)
    for source in base.glob('*.json'):
        if not (destination/source.name).exists():shutil.copy2(source,destination/source.name)
    write_json(marker,{'adapter_sha256':adapter_hash,'base_revision':source_spec['revision']})
    del model
    return destination,adapter_hash

def calibration_text(root):
    from streamlined.common import atomic
    root=Path(root);rows=calibration_rows(root)
    text='\n\n'.join(row['prompt_text'] for row in rows)+'\n'
    target=root/'data/calibration'/f'{CALIBRATION_NAME}.txt'
    sha=digest(text)
    if target.exists() and digest(target.read_bytes())!=sha:
        raise ValueError('Calibration text differs from frozen 604 prompt selection')
    if not target.exists():atomic(target,text)
    write_json(root/'data/calibration'/f'{CALIBRATION_NAME}.json',
        {'prompt_ids':[row['prompt_id'] for row in rows], 'sha256':sha,
         'prompt_hashes':[row['prompt_sha256'] for row in rows],
         'delimiter':'two newline characters; final newline'})
    return target,sha

def artifact_paths(root,key,representation,adapter=None):
    stage=condition_key(key,representation,adapter)
    folder=Path(root)/'models/quantized'/key/stage/representation
    return folder,folder/'model.gguf',folder/'artifact.json'

def tensor_types(artifact,toolkit,representation):
    """Check actual GGUF tensors, not just the requested quantizer flag."""
    directory=toolkit['repo']/('3rdparty/llama.cpp/gguf-py' if representation=='native_i2_s' else 'gguf-py')
    if not directory.is_dir():raise RuntimeError(f'Pinned GGUF Python reader missing: {directory}')
    # Inspect in a clean interpreter so an earlier Transformers/PEFT import
    # cannot substitute another installed ``gguf`` module for the pinned one.
    script='''
import json
from collections import Counter
import gguf
reader=gguf.GGUFReader(__import__("sys").argv[1])
counts=Counter(gguf.GGMLQuantizationType(int(t.tensor_type)).name for t in reader.tensors)
print("CAM_LDS_TENSOR_TYPES="+json.dumps(dict(counts),sort_keys=True))
'''
    env=os.environ.copy()
    env['PYTHONPATH']=str(directory)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result=subprocess.run([sys.executable,'-c',script,str(artifact)],cwd=artifact.parent,
                          env=env,text=True,capture_output=True,check=False)
    if result.returncode:
        detail=(result.stderr or result.stdout)[-2000:]
        raise RuntimeError(f'Pinned GGUF tensor audit failed: {detail}')
    marker='CAM_LDS_TENSOR_TYPES='
    lines=[line for line in result.stdout.splitlines() if line.startswith(marker)]
    if not lines:
        raise RuntimeError(f'Pinned GGUF tensor audit returned no result: {result.stdout[-2000:]}')
    counts=Counter(json.loads(lines[-1][len(marker):]))
    target={'native_i2_s':'I2_S','tq1_0':'TQ1_0',
            'iq1_s_camlds_calibrated':'IQ1_S'}[representation]
    if counts[target]<1:
        raise ValueError(f'GGUF contains no {target} tensors; actual types: {dict(counts)}')
    return dict(counts)

def prepare(root,key,representation,adapter,mirror):
    from huggingface_hub import hf_hub_download
    root=Path(root);verify_experiment(root)
    folder,artifact,manifest=artifact_paths(root,key,representation,adapter)
    pending=folder/'artifact.pending.json'
    folder.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():mirror.fetch(manifest.relative_to(root).as_posix(),manifest)
    if manifest.exists():
        saved=read_json(manifest)
        if not artifact.exists():mirror.fetch(artifact.relative_to(root).as_posix(),artifact)
        if artifact.exists() and file_hash(artifact)==saved['artifact_sha256']:
            mirror.put(root,manifest)
            return saved
        raise ValueError('Saved GGUF artifact differs from its manifest')
    if pending.exists() and artifact.exists():
        saved=read_json(pending)
        if file_hash(artifact)!=saved.get('artifact_sha256'):
            raise ValueError('Prepared but unacknowledged GGUF differs from pending manifest')
        mirror.put(root,artifact)
        write_json(manifest,saved);mirror.put(root,manifest)
        pending.unlink()
        return saved
    started=time.perf_counter()
    toolkit=toolchain(root,bitnet=(key=='bitnet2b'),mirror=mirror)
    source_revision=read_json(root/'manifests/models.json')[key]['revision']
    calibration_sha='';imatrix_sha='';adapter_hash='';source_kind='frozen_hf'
    if key=='bitnet2b' and adapter is None:
        downloaded=hf_hub_download(BITNET_GGUF_REPO,BITNET_GGUF_FILE,
                                   revision=BITNET_GGUF_REVISION)
        if file_hash(downloaded)!=BITNET_GGUF_SHA256:
            raise ValueError('Official BitNet I2_S file hash differs from pinned artifact')
        shutil.copy2(downloaded,artifact)
        source_kind='official_bitnet_i2_s'
    else:
        hf_dir,adapter_hash=merged_snapshot(root,key,adapter,mirror)
        if key=='bitnet2b':
            converter=toolkit['repo']/'utils/convert-hf-to-gguf-bitnet.py'
            # A verifier failure can leave a complete adapter-specific GGUF
            # before artifact.pending.json is created.  Reuse it only after
            # the pinned reader proves that it is a valid I2_S artifact.
            recovered=False
            if artifact.exists() and artifact.stat().st_size>=1000000:
                try:
                    with artifact.open('rb') as handle: recovered=handle.read(4)==b'GGUF'
                    if recovered:tensor_types(artifact,toolkit,representation)
                except Exception:
                    recovered=False
            if not recovered:
                run([sys.executable,converter,hf_dir,'--outtype','i2_s','--outfile',artifact],
                    root,f'{key}_{adapter}_i2s_convert.log')
            source_kind='merged_bitnet_bf16_master'
        else:
            intermediate=root/'.cache/gguf_intermediate'/key/(adapter or 'untreated')/'bf16.gguf'
            intermediate.parent.mkdir(parents=True,exist_ok=True)
            if not intermediate.exists():
                run([sys.executable,toolkit['repo']/'convert_hf_to_gguf.py',hf_dir,
                     '--outtype','bf16','--outfile',intermediate],
                    root,f'{key}_{adapter or "untreated"}_convert.log')
            command=[toolkit['quantizer']]
            if representation=='iq1_s_camlds_calibrated':
                corpus,calibration_sha=calibration_text(root)
                matrix=root/'.cache/imatrix'/key/(adapter or 'untreated')/'imatrix.gguf'
                matrix.parent.mkdir(parents=True,exist_ok=True)
                if not matrix.exists():
                    run([toolkit['imatrix'],'-m',intermediate,'-f',corpus,'-o',matrix,
                         '-ngl','999','-c',str(min(8192,read_json(root/'manifests/models.json')[key]['context']))],
                        root,f'{key}_{adapter or "untreated"}_imatrix.log')
                imatrix_sha=file_hash(matrix)
                command.extend(['--imatrix',matrix])
            command.extend([intermediate,artifact,
                            'IQ1_S' if representation=='iq1_s_camlds_calibrated' else 'TQ1_0'])
            run(command,root,f'{key}_{adapter or "untreated"}_{representation}_quantize.log')
    if not artifact.exists() or artifact.stat().st_size<1000000:
        raise RuntimeError('GGUF output missing or implausibly small')
    with artifact.open('rb') as handle:
        if handle.read(4)!=b'GGUF':raise ValueError('Artifact lacks GGUF magic header')
    quantization_tensor_counts=tensor_types(artifact,toolkit,representation)
    saved=dict(model=key,representation=representation,adapter=adapter,
        artifact_relative=artifact.relative_to(root).as_posix(),artifact_sha256=file_hash(artifact),
        artifact_bytes=artifact.stat().st_size,source_revision=source_revision,
        source_kind=source_kind,adapter_sha256=adapter_hash,
        calibration_sha256=calibration_sha,imatrix_sha256=imatrix_sha,
        calibration_name=CALIBRATION_NAME if calibration_sha else '',
        quantization_tensor_counts=quantization_tensor_counts,
        toolchain_commit=toolkit['commit'],converter_patch=toolkit.get('converter_patch',''),
        preparation_seconds=time.perf_counter()-started)
    # The file is acknowledged on the laptop before the manifest marks it usable.
    write_json(pending,saved)
    mirror.put(root,artifact)
    write_json(manifest,saved);mirror.put(root,manifest)
    pending.unlink()
    return saved
