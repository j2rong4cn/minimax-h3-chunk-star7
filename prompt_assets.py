"""Per-user prompt assets, written atomically outside the installed node sources."""
import json
import os
from pathlib import Path
import re
import tempfile
import threading

_LOCK = threading.RLock()
_MAX_BYTES = 4 * 1024 * 1024
# User-supplied ORB360 prompts, kept verbatim with short Chinese display titles.
# Source: https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin
_SEEDS = Path(__file__).with_name('prompt_presets.json')


class ConflictError(ValueError):
    pass


def validate_entries(entries):
    if not isinstance(entries, list) or len(entries) > 500:
        raise ValueError('提示词存档数量无效（最多 500 条）')
    seen = set()
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('提示词存档格式无效')
        key, title, text = (entry.get(name) for name in ('id', 'title', 'text'))
        if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', key) or key in seen:
            raise ValueError('提示词存档标识无效或重复')
        if not isinstance(title, str) or not title.strip() or len(title) > 120:
            raise ValueError('标题不能为空且不能超过 120 字')
        if not isinstance(text, str) or not text.strip() or len(text) > 250_000 or '\x00' in text:
            raise ValueError('提示词不能为空且不能超过 250000 字')
        seen.add(key)
        result.append({'id': key, 'title': title.strip(), 'text': text})
    return result


class PromptAssetStore:
    def __init__(self, path, seeds=_SEEDS):
        self.path = Path(path)
        self.seeds = Path(seeds)

    def read(self):
        with _LOCK:
            if not self.path.exists():
                return {'revision': 0, 'entries': validate_entries(json.loads(self.seeds.read_text(encoding='utf-8')))}
            if self.path.stat().st_size > _MAX_BYTES:
                raise ValueError('提示词存档文件超过容量限制')
            value = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(value, dict) or type(value.get('revision')) is not int or value['revision'] < 0:
                raise ValueError('提示词存档文件格式无效')
            return {'revision': value['revision'], 'entries': validate_entries(value.get('entries'))}

    def write(self, revision, entries):
        entries = validate_entries(entries)
        with _LOCK:
            current = self.read()
            if type(revision) is not int or revision != current['revision']:
                raise ConflictError('提示词库已被其他窗口更新，请重新打开；当前编辑内容仍保留，可复制后再保存。')
            result = {'revision': revision + 1, 'entries': entries}
            payload = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
            if len(payload) > _MAX_BYTES:
                raise ValueError('提示词库总容量超过 4 MiB')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.prompt-assets-', suffix='.tmp', dir=self.path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return result


def register_routes():
    from aiohttp import web
    from server import PromptServer

    def store(request):
        path = PromptServer.instance.user_manager.get_request_user_filepath(
            request, 'star7/prompt-assets.json', create_dir=False,
        )
        if path is None:
            raise ValueError('当前用户的存档位置不可用')
        return PromptAssetStore(path)

    @PromptServer.instance.routes.get('/minimax-h3-chunk-star7/prompt-assets')
    async def read_assets(request):
        try:
            return web.json_response(store(request).read())
        except (ValueError, OSError, KeyError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @PromptServer.instance.routes.post('/minimax-h3-chunk-star7/prompt-assets')
    async def save_assets(request):
        try:
            if request.content_length is not None and request.content_length > _MAX_BYTES:
                raise ValueError('提示词库总容量超过 4 MiB')
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError('提示词存档格式无效')
            return web.json_response(store(request).write(value.get('revision'), value.get('entries')))
        except ConflictError as error:
            return web.json_response({'error': str(error)}, status=409)
        except (ValueError, OSError, KeyError) as error:
            return web.json_response({'error': str(error)}, status=400)
