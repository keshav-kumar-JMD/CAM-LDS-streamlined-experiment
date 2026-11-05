from pathlib import Path

from continuation.artifacts import BITNET_CONVERTER_PATCH,cmake_commands,patch_bitnet_converter


def test_bitnet_uses_official_server_options_and_server_only_build():
    configure,build=cmake_commands(Path('/src'),Path('/build'),'80',bitnet=True,jobs=6)
    assert '-DLLAMA_BUILD_TOOLS=ON' in configure
    assert '-DLLAMA_BUILD_EXAMPLES=ON' in configure
    assert '-DLLAMA_BUILD_COMMON=ON' in configure
    assert '-DLLAMA_BUILD_SERVER=ON' in configure
    assert build[-2:]==['--target','llama-server']


def test_llama_cpp_still_builds_selected_targets():
    _,build=cmake_commands(Path('/src'),Path('/build'),'80',bitnet=False,jobs=6)
    assert build[-4:]==['--target','llama-server','llama-quantize','llama-imatrix']


def test_bitnet_converter_multishard_names_are_reusable(tmp_path):
    converter=tmp_path/'utils/convert-hf-to-gguf-bitnet.py'
    converter.parent.mkdir()
    converter.write_text('''
def names(self):
    return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))
def bins(self):
    return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))
''',encoding='utf-8')
    assert patch_bitnet_converter(tmp_path)==BITNET_CONVERTER_PATCH
    patched=converter.read_text(encoding='utf-8')
    assert 'return tuple(f"model-' in patched
    assert 'return tuple(f"pytorch_model-' in patched
    # A repeated call is idempotent and continues to validate the layout.
    assert patch_bitnet_converter(tmp_path)==BITNET_CONVERTER_PATCH
