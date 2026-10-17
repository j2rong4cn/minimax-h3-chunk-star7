"""Platform dispatch and cross-platform manifest preservation, without a compiler."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest import mock, TestCase

ROOT = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_platform_entries_preserve_legacy_windows_and_reject_wrong_scale():
    pre = load(ROOT / "h3_preprocess.py", "preprocess_packaging_test")
    windows = dict(file="win_amd64/test.dll", sha256="old")
    linux = dict(file="linux_x86_64/test.so", sha256="new")
    assert pre._manifest_entry(windows, "win32", "AMD64") is windows
    manifest = dict(windows_x64=windows, linux_x86_64=linux, swiglu_divisor=256)
    assert pre._manifest_entry(manifest, "win32", "AMD64") is windows
    assert pre._manifest_entry(manifest, "linux", "x86_64") is linux
    for system, machine in (("darwin", "x86_64"), ("linux", "aarch64")):
        with TestCase().assertRaisesRegex(RuntimeError, "No H3 preprocessing"):
            pre._manifest_entry(manifest, system, machine)
    with TestCase().assertRaisesRegex(RuntimeError, "contract mismatch"):
        pre._manifest_entry(dict(manifest, swiglu_divisor=128), "linux", "x86_64")
    with TestCase().assertRaisesRegex(RuntimeError, "No bundled"):
        pre._manifest_entry(windows, "linux", "x86_64")


def test_linux_builder_keeps_windows_checksum_and_original_scale():
    builder = load(ROOT / "csrc/build_linux_native.py", "linux_builder_test")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "bin").mkdir()
        manifest = root / "bin/h3_preprocess_manifest.json"
        manifest.write_text(json.dumps(dict(abi_version=1, file="win_amd64/original.dll", sha256="original")))
        commands = []

        def compiler(command, **kwargs):
            commands.append(command)
            Path(command[command.index("-o") + 1]).write_bytes(b"test library")

        with mock.patch.object(builder, "ROOT", root), mock.patch.object(builder.subprocess, "run", side_effect=compiler):
            builder.build("preprocess", "/usr/local/cuda/bin/nvcc")
        data = json.loads(manifest.read_text())
        assert data["windows_x64"] == dict(file="win_amd64/original.dll", sha256="original")
        assert data["swiglu_divisor"] == 256
        assert data["linux_x86_64"]["file"].endswith(".so")
        assert "--use_fast_math" not in commands[0]
        assert "GPU numerical validation pending" in data["linux_x86_64"]["validation"]
