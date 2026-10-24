import test from 'node:test';
import assert from 'node:assert/strict';
import { uniqueTitle, readTextFiles, currentPromptTitle } from './web/prompt-assets.mjs';

test('current prompt names use a short local timestamp and numbered collisions', () => {
    const title = currentPromptTitle(new Date(2026, 9, 21, 4, 20));
    assert.equal(title, '261021-0420提示词');
    assert.equal(uniqueTitle(title, [{ title }, { title: title + '（2）' }]), title + '（3）');
});

test('batch TXT retains complete content and assigns distinct duplicate titles', async () => {
    const current = [{ title: '镜头' }];
    const files = [new File(['\uFEFF正文\n<Picture 1>\n'], '镜头.txt'), new File(['第二份'], '镜头.TXT'), new File(['{}'], 'workflow.json')];
    const { imported, ignored } = await readTextFiles(files, current);
    assert.deepEqual(imported.map(entry => entry.title), ['镜头（2）', '镜头（3）']);
    assert.equal(imported[0].text, '正文\n<Picture 1>\n');
    assert.equal(new Set(imported.map(entry => entry.id)).size, 2);
    assert.deepEqual(ignored, ['workflow.json']);
    assert.deepEqual(current, [{ title: '镜头' }]);
});

test('invalid batch rejects without modifying existing entries', async () => {
    const current = [{ title: '保留' }];
    await assert.rejects(readTextFiles([new File(['okay'], 'a.txt'), new File([''], 'b.txt')], current));
    await assert.rejects(readTextFiles([new File(['a\0b'], 'a.txt')], current));
    assert.deepEqual(current, [{ title: '保留' }]);
});

test('Chinese legacy text decodes and title stays within storage limits', async () => {
    const { imported } = await readTextFiles([new File([new Uint8Array([0xd6, 0xd0, 0xce, 0xc4])], 'GBK.txt')], []);
    assert.equal(imported[0].text, '中文');
    const utf16 = await readTextFiles([new File([new Uint8Array([0xff, 0xfe, 0x2d, 0x4e, 0x87, 0x65])], 'Unicode.txt')], []);
    assert.equal(utf16.imported[0].text, '中文');
    assert.ok(uniqueTitle('长'.repeat(120), []).length <= 120);
});
