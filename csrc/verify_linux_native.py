"""Host-only ELF/ABI checks; explicitly does not claim GPU validation."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    if sys.platform != "linux":
        raise SystemExit("Linux host required")
    expected = {
        "h3_preprocess": ("star7_h3_preprocess_abi", None, None),
        "veda_sm75": ("star7_veda_abi_version", "star7_veda_args_size", 192),
        "w4a8": ("star7_w4a8_abi_version", "star7_w4a8_args_size", 104),
    }
    records = []
    for name, (abi, args, size) in expected.items():
        manifest = json.loads((ROOT / "bin" / (name + "_manifest.json")).read_text())
        entry = manifest["linux_x86_64"]
        path = ROOT / "bin" / entry["file"]
        data = path.read_bytes()
        assert data[:6] == b"\x7fELF\x02\x01" and int.from_bytes(data[18:20], "little") == 62
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        dependencies = subprocess.check_output(["ldd", str(path)], text=True)
        assert "not found" not in dependencies
        assert not any(word in dependencies for word in ("libcudart", "libtorch", "libpython", "libstdc++"))
        lib = ctypes.CDLL(str(path))
        fn = getattr(lib, abi)
        fn.argtypes, fn.restype = [], ctypes.c_int
        assert fn() == 1
        if args:
            fn = getattr(lib, args)
            fn.argtypes, fn.restype = [], ctypes.c_int
            assert fn() == size
        records.append({"file": entry["file"], "sha256": entry["sha256"],
                        "host_abi": "pass", "dependencies": dependencies})
    report = {"gpu_tested": False, "libraries": records}
    (ROOT / "bin/linux_x86_64/build-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
