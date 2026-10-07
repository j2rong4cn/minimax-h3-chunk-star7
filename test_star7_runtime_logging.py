"""Own log prefixes and safe fallback when the W4A8 C ABI library cannot load."""
import logging
import pathlib
import sys
import types
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
pkg = types.ModuleType('s7logs'); pkg.__path__ = [str(ROOT)]; sys.modules[pkg.__name__] = pkg
from s7logs.vendor.veda.status import NodeStatus
from s7logs import star7_w4a8

records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record)

logger = logging.getLogger('Star7-H3-VEDA')
handler = Capture(); logger.addHandler(handler)
old_level = logger.level; logger.setLevel(logging.INFO)
try:
    status = NodeStatus('test')
    status.show('Running · SM75\nVideo: 1088x1920\n')
    status.show('Running · SM75\nVideo: 1088x1920\n')
    status.warn('DLL error: 找不到指定的模块')
    assert len(records) == 3
    assert all(r.getMessage().startswith('[Star7 H3 VEDA] ') for r in records)
    assert records[0].getMessage() == '[Star7 H3 VEDA] Running | SM75'
    assert '找不到指定的模块' in records[-1].getMessage()
    assert records[-1].levelno == logging.WARNING
    assert not logger.handlers[:-1], 'test must not install a duplicate production handler'
finally:
    logger.removeHandler(handler); logger.setLevel(old_level)

star7_w4a8._LOAD_ATTEMPTED = False
with patch.object(star7_w4a8.w4a8_native, 'Kernel', side_effect=OSError('DLL unavailable')):
    assert star7_w4a8._load_kernel() is None
star7_w4a8._LOAD_ATTEMPTED = False
assert star7_w4a8._load_kernel() is not None
print('Star7 multiline prefix / repeat suppression / diagnostic text / W4A8 library fallback: PASS')
