import json
import logging
import platform

LOG = logging.getLogger('Star7SparkVPack')


def hardware():
    import torch
    info = {'gpu': 'CPU', 'capability': None, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'os': platform.system()}
    try:
        import comfyui_version
        info['comfyui'] = comfyui_version.__version__
    except ImportError:
        info['comfyui'] = 'unknown'
    if torch.cuda.is_available():
        from comfy.model_management import get_torch_device
        device = get_torch_device()
        if device.type == 'cuda':
            info.update(gpu=torch.cuda.get_device_name(device), capability=tuple(torch.cuda.get_device_capability(device)))
    return info


def emit(node_id, report):
    LOG.info('Star7 Spark/VPack | %s', json.dumps(report, ensure_ascii=False, default=str))
    if node_id is None:
        return
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.send_sync('star7_h3_unified_status', {'node_id': str(node_id), 'report': report})
    except Exception:
        # Telemetry must never fail an otherwise valid model evaluation.
        LOG.debug('Spark/VPack UI event unavailable', exc_info=True)


def text(report):
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
