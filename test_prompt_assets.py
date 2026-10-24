"""Prompt-library persistence checks; all writes use disposable test folders."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prompt_assets_tests', ROOT / 'prompt_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'user' / 'prompt-assets.json'
        self.store = assets.PromptAssetStore(self.path)

    def test_presets_have_chinese_titles_and_preserve_supplied_text(self):
        initial = self.store.read()
        self.assertEqual(initial['revision'], 0)
        self.assertEqual([entry['title'] for entry in initial['entries']], ['实景环绕', '三图环绕', '仰视环绕', '跨高度环绕'])
        self.assertFalse(self.path.exists())
        for entry in initial['entries']:
            self.assertIn('ORB360_CW', entry['text'])
            self.assertIn('non_diegetic_music:', entry['text'])

    def test_save_rename_duplicate_delete_and_empty_library_survive_restart(self):
        entry = {'id': 'example', 'title': '我的提示词', 'text': '完整内容\n<Picture 1>\n'}
        state = self.store.write(0, [entry])
        entry = {**entry, 'title': '已重命名'}
        copy = {**entry, 'id': 'copy', 'title': '已重命名（2）'}
        state = self.store.write(state['revision'], [entry, copy])
        reopened = assets.PromptAssetStore(self.path).read()
        self.assertEqual(reopened, state)
        self.assertEqual(reopened['entries'][1]['text'], entry['text'])
        self.store.write(state['revision'], [])
        self.assertEqual(assets.PromptAssetStore(self.path).read()['entries'], [])

    def test_stale_window_cannot_overwrite_newer_edits(self):
        first = self.store.read()
        saved = self.store.write(first['revision'], first['entries'])
        with self.assertRaises(assets.ConflictError):
            self.store.write(first['revision'], [])
        self.assertEqual(self.store.read(), saved)

    def test_invalid_batch_is_atomic_and_never_overwrites_existing_file(self):
        state = self.store.read()
        self.store.write(0, state['entries'])
        before = self.path.read_bytes()
        for bad in ([{'id': '../bad', 'title': 'x', 'text': 'x'}], [state['entries'][0]] * 2,
                    [{'id': 'a', 'title': '', 'text': 'x'}], [{'id': 'a', 'title': 'x', 'text': ''}],
                    [{'id': 'a', 'title': 'x', 'text': 'x' * 250001}]):
            with self.assertRaises(ValueError):
                self.store.write(1, bad)
            self.assertEqual(self.path.read_bytes(), before)

    def test_failed_atomic_replace_preserves_previous_file_and_cleans_temporary(self):
        state = self.store.read()
        self.store.write(0, state['entries'])
        before = self.path.read_bytes()
        with patch.object(assets.os, 'replace', side_effect=OSError('simulated failure')):
            with self.assertRaises(OSError):
                self.store.write(1, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_corrupt_file_is_reported_without_replacing_it_with_presets(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('invalid', encoding='utf-8')
        with self.assertRaises(ValueError): self.store.read()
        with self.assertRaises(ValueError): self.store.write(0, [])
        self.assertEqual(self.path.read_text(), 'invalid')


class Routes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        routes = web.RouteTableDef()
        def user_path(request, filename, create_dir=False):
            user = request.headers.get('comfy-user', 'default')
            if user not in ('default', 'second'): raise KeyError('Unknown user')
            return Path(self.temp.name) / user / filename
        server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(
            routes=routes, user_manager=types.SimpleNamespace(get_request_user_filepath=user_path),
        )))
        with patch.dict(sys.modules, {'server': server}): assets.register_routes()
        app = web.Application(); app.add_routes(routes)
        self.client = TestClient(TestServer(app)); await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_http_roundtrip_and_users_are_isolated(self):
        endpoint = '/minimax-h3-chunk-star7/prompt-assets'
        response = await self.client.get(endpoint); initial = await response.json()
        self.assertEqual(response.status, 200)
        entry = {'id': 'custom', 'title': 'HTTP 存档', 'text': '<Picture 1> 原始正文'}
        response = await self.client.post(endpoint, json={'revision': 0, 'entries': [entry]})
        self.assertEqual(response.status, 200)
        response = await self.client.get(endpoint)
        self.assertEqual((await response.json())['entries'], [entry])
        response = await self.client.get(endpoint, headers={'comfy-user': 'second'})
        self.assertEqual((await response.json()), initial)
        response = await self.client.post(endpoint, json={'revision': 0, 'entries': []})
        self.assertEqual(response.status, 409)
        response = await self.client.get(endpoint)
        self.assertEqual((await response.json())['entries'], [entry])

    async def test_malformed_request_does_not_create_user_file(self):
        endpoint = '/minimax-h3-chunk-star7/prompt-assets'
        for payload in ([], {'revision': 0, 'entries': [{'title': 'bad'}]}):
            response = await self.client.post(endpoint, json=payload)
            self.assertEqual(response.status, 400)
        self.assertFalse(list(Path(self.temp.name).rglob('*.json')))


if __name__ == '__main__':
    unittest.main()
