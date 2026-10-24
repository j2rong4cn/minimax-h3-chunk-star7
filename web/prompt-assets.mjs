const ENDPOINT = '/minimax-h3-chunk-star7/prompt-assets';
let active = null;

export function currentPromptTitle(date = new Date()) {
    const pad = value => String(value).padStart(2, '0');
    return `${pad(date.getFullYear() % 100)}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}提示词`;
}

export function uniqueTitle(title, entries) {
    const base = String(title).trim().slice(0, 110) || '提示词';
    const names = new Set(entries.map(entry => entry.title));
    if (!names.has(base)) return base;
    let ordinal = 2;
    while (names.has(`${base}（${ordinal}）`)) ordinal++;
    return `${base}（${ordinal}）`;
}

export async function readTextFiles(files, entries) {
    const imported = [], ignored = [];
    for (const file of files) {
        if (!/\.txt$/i.test(file.name)) { ignored.push(file.name); continue; }
        if (file.size > 1024 * 1024) throw Error(`文件过大：${file.name}`);
        const bytes = await file.arrayBuffer();
        const prefix = new Uint8Array(bytes, 0, Math.min(bytes.byteLength, 2));
        let text;
        if (prefix[0] === 0xff && prefix[1] === 0xfe) text = new TextDecoder('utf-16le', { fatal: true }).decode(bytes);
        else if (prefix[0] === 0xfe && prefix[1] === 0xff) text = new TextDecoder('utf-16be', { fatal: true }).decode(bytes);
        else {
            try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
            catch { text = new TextDecoder('gb18030', { fatal: true }).decode(bytes); }
        }
        text = text.replace(/^\uFEFF/, '');
        if (!text.trim() || text.length > 250000 || text.includes('\0')) throw Error(`文本为空或内容无效：${file.name}`);
        const filename = file.name.split(/[\\/]/).pop().replace(/\.txt$/i, '');
        imported.push({ id: crypto.randomUUID(), title: uniqueTitle(filename, [...entries, ...imported]), text });
    }
    return { imported, ignored };
}

function style() {
    if (document.getElementById('star7-pa-style')) return;
    const sheet = document.createElement('style'); sheet.id = 'star7-pa-style';
    sheet.textContent = `
.star7-pa{box-sizing:border-box;width:min(1080px,94vw);height:min(760px,88vh);max-width:none;max-height:92vh;padding:0;border:1px solid #555b68;border-radius:14px;background:var(--comfy-menu-bg,#20242d);color:var(--fg-color,#ececf3);box-shadow:0 20px 80px #0008;font:17px/1.5 sans-serif;overflow:hidden}
.star7-pa::backdrop{background:#0009}.star7-pa *{box-sizing:border-box}.star7-pa .pa-shell{display:flex;height:100%;flex-direction:column}
.star7-pa button,.star7-pa input,.star7-pa textarea{font:inherit;color:inherit;border:1px solid #555b68;border-radius:7px;background:var(--comfy-input-bg,#171b23);padding:8px 12px;margin:0;min-width:0}
.star7-pa button{cursor:pointer;white-space:nowrap;background:var(--comfy-input-bg,#2a303c)}.star7-pa button:hover{border-color:#a99bed}.star7-pa button:disabled{opacity:.45;cursor:default}.star7-pa .pa-primary{background:#7761ca;border-color:#9682e6;color:white}.star7-pa .pa-danger{color:#f4a799}.star7-pa :focus-visible{outline:2px solid #af9aff;outline-offset:2px}
.star7-pa .pa-header{display:flex;gap:12px;align-items:center;padding:16px 20px;border-bottom:1px solid #414651}.star7-pa h2{font-size:20px;margin:0;flex:1}.star7-pa .pa-body{display:grid;grid-template-columns:260px minmax(0,1fr);flex:1;min-height:0}
.star7-pa .pa-sidebar{display:flex;flex-direction:column;gap:10px;padding:16px;border-right:1px solid #414651;min-height:0}.star7-pa .pa-list{overflow:auto;flex:1;display:flex;flex-direction:column;gap:6px}.star7-pa .pa-item{display:block;text-align:left;width:100%;white-space:normal;overflow-wrap:anywhere;padding:10px 12px}.star7-pa .pa-item[aria-selected=true]{background:#7561b333;border-color:#a18bdc}.star7-pa .pa-count{color:#acb2c1}.star7-pa .pa-empty{color:#acb2c1;padding:12px 0}
.star7-pa .pa-editor{display:flex;flex-direction:column;gap:12px;padding:16px 20px;min-width:0;min-height:0}.star7-pa .pa-title{width:100%}.star7-pa .pa-text{flex:1;resize:none;width:100%;min-height:100px;line-height:1.65;tab-size:2}.star7-pa .pa-tools{display:flex;flex-wrap:wrap;gap:8px}.star7-pa .pa-status{min-height:48px;padding:10px 20px;border-top:1px solid #414651;color:#b9aedb;overflow-wrap:anywhere}.star7-pa .pa-status[data-error=true]{color:#f4a799}.star7-pa .pa-drop{border:1px dashed #747083;border-radius:8px;padding:10px;text-align:center;color:#c4bdd7}.star7-pa.pa-dragging .pa-drop{background:#7761ca33;border-color:#b39aff}.star7-pa .pa-confirm{display:flex;flex-wrap:wrap;align-items:center;gap:8px;color:#f4a799}.star7-pa [hidden]{display:none!important}
@media(max-width:700px){.star7-pa .pa-header{flex-wrap:wrap;padding:12px}.star7-pa h2{flex-basis:70%}.star7-pa .pa-body{grid-template-columns:1fr;grid-template-rows:190px minmax(0,1fr)}.star7-pa .pa-sidebar{border-right:0;border-bottom:1px solid #414651;padding:10px}.star7-pa .pa-sidebar .pa-drop{display:none}.star7-pa .pa-editor{padding:12px}.star7-pa .pa-header button{padding:6px 9px}}
`;
    document.head.append(sheet);
}

export async function openPromptAssets({ fetchApi, getCurrent, apply }) {
    if (active?.open) { active.focus(); return; }
    active?.remove();
    style();
    const previousFocus = document.activeElement;
    const dialog = document.createElement('dialog');
    dialog.className = 'star7-pa'; dialog.dataset.star7PromptAssets = '';
    dialog.setAttribute('aria-label', '资产提示词');
    dialog.innerHTML = `<div class="pa-shell"><header class="pa-header"><h2>资产提示词</h2><button data-action="capture">保存当前提示词</button><button data-action="import">导入 TXT</button><button data-action="close" aria-label="关闭资产提示词">关闭</button></header><div class="pa-body"><aside class="pa-sidebar"><input class="pa-search" type="search" placeholder="搜索提示词" aria-label="搜索提示词"><button data-action="new">＋ 新建提示词</button><div class="pa-list" role="listbox" aria-label="已保存的提示词"></div><span class="pa-count"></span><div class="pa-drop">拖入 TXT · 支持批量</div></aside><section class="pa-editor"><input class="pa-title" maxlength="120" placeholder="存档名称" aria-label="存档名称"><textarea class="pa-text" placeholder="在此编写或粘贴提示词" aria-label="存档提示词" spellcheck="false"></textarea><div class="pa-tools"><button class="pa-primary" data-action="use">使用</button><button data-action="save">保存修改</button><button data-action="duplicate">存为副本</button><button class="pa-danger" data-action="delete">删除</button></div><div class="pa-confirm" hidden><span>删除此存档？</span><button data-action="delete-cancel">取消</button><button class="pa-danger" data-action="delete-confirm">确认删除</button></div></section></div><footer class="pa-status" role="status" aria-live="polite">正在读取提示词库…</footer><input class="pa-file" type="file" accept=".txt,text/plain" multiple hidden></div>`;
    const $ = selector => dialog.querySelector(selector);
    const title = $('.pa-title'), body = $('.pa-text'), list = $('.pa-list'), search = $('.pa-search');
    let snapshot = null, selected = null, baseline = '', busy = false;
    const fingerprint = () => JSON.stringify([title.value, body.value]);
    const dirty = () => fingerprint() !== baseline;
    const status = (message, error = false) => { $('.pa-status').textContent = message; $('.pa-status').dataset.error = String(error); };
    const setBusy = value => { busy = value; for (const button of dialog.querySelectorAll('button')) button.disabled = value; title.readOnly = body.readOnly = value; search.disabled = value; };
    const canLeave = () => !dirty() || window.confirm('当前更改尚未保存，是否放弃这些更改？');
    const close = () => { if (!busy && canLeave()) dialog.close(); };
    const outside = event => {
        const bounds = dialog.getBoundingClientRect();
        return event.clientX < bounds.left || event.clientX > bounds.right
            || event.clientY < bounds.top || event.clientY > bounds.bottom;
    };
    let pressedOutside = false;
    dialog.addEventListener('pointerdown', event => {
        pressedOutside = event.target === dialog && outside(event);
    });
    const choose = entry => {
        selected = entry?.id || null; title.value = entry?.title || ''; body.value = entry?.text || '';
        baseline = fingerprint(); $('.pa-confirm').hidden = true; render();
    };
    function render() {
        list.replaceChildren();
        const query = search.value.trim().toLowerCase();
        const visible = (snapshot?.entries || []).filter(entry => `${entry.title}\n${entry.text}`.toLowerCase().includes(query));
        for (const entry of visible) {
            const button = document.createElement('button'); button.className = 'pa-item'; button.type = 'button';
            button.setAttribute('role', 'option'); button.setAttribute('aria-selected', String(selected === entry.id));
            button.textContent = entry.title; button.title = entry.title; button.disabled = busy;
            button.onclick = () => { if (!busy && canLeave()) choose(entry); }; list.append(button);
        }
        if (!visible.length) { const empty = document.createElement('span'); empty.className = 'pa-empty'; empty.textContent = query ? '没有匹配的提示词' : '还没有保存的提示词'; list.append(empty); }
        $('.pa-count').textContent = `共 ${snapshot?.entries.length || 0} 条`;
        $('[data-action=delete]').disabled = busy || !snapshot?.entries.some(entry => entry.id === selected);
    }
    const readEditor = () => {
        if (!title.value.trim()) throw Error('请填写存档名称');
        if (!body.value.trim() || body.value.length > 250000) throw Error('提示词不能为空且不能超过 250000 字');
        return { id: selected || crypto.randomUUID(), title: title.value.trim(), text: body.value };
    };
    const persist = async entries => {
        const response = await fetchApi(ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ revision: snapshot.revision, entries }) });
        const result = await response.json();
        if (!response.ok) throw Error(result.error || '保存失败');
        snapshot = result;
    };
    const operation = async callback => {
        if (busy || !snapshot) return;
        setBusy(true);
        try { await callback(); }
        catch (error) { status(error.message || String(error), true); }
        finally { setBusy(false); render(); }
    };
    const importFiles = async files => {
        if (busy || !snapshot || !canLeave()) return;
        await operation(async () => {
            const { imported, ignored } = await readTextFiles(files, snapshot.entries);
            if (!imported.length) throw Error('请拖入 TXT 文本文件');
            await persist([...snapshot.entries, ...imported]); choose(imported[0]);
            status(`已保存 ${imported.length} 条提示词${ignored.length ? `，已跳过 ${ignored.length} 个非 TXT 文件` : ''}`);
        });
    };
    dialog.addEventListener('click', event => {
        // Only a click that starts and ends on the backdrop dismisses the panel.
        // Dragging a text selection outside must not close the editor.
        if (event.target === dialog && pressedOutside && outside(event)) {
            pressedOutside = false; close(); return;
        }
        pressedOutside = false;
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (!action || busy) return;
        if (action === 'close') return close();
        if (!snapshot) return;
        if (action === 'import') return $('.pa-file').click();
        if (action === 'new') { if (canLeave()) { choose({ id: crypto.randomUUID(), title: uniqueTitle('新提示词', snapshot.entries), text: '' }); title.focus(); title.select(); } return; }
        if (action === 'delete') { $('.pa-confirm').hidden = false; return; }
        if (action === 'delete-cancel') { $('.pa-confirm').hidden = true; return; }
        if (action === 'use') { if (!body.value.trim()) return status('提示词为空', true); apply(body.value); baseline = fingerprint(); dialog.close(); return; }
        void operation(async () => {
            if (action === 'capture') {
                if (!canLeave()) return;
                const text = String(getCurrent() || '');
                if (!text.trim()) throw Error('当前节点的提示词为空');
                const entry = { id: crypto.randomUUID(), title: uniqueTitle(currentPromptTitle(), snapshot.entries), text };
                await persist([...snapshot.entries, entry]); choose(entry); title.focus(); title.select(); status('已保存当前提示词，可修改名称后保存');
            } else if (action === 'save') {
                const entry = readEditor(), entries = snapshot.entries.filter(item => item.id !== entry.id);
                const index = snapshot.entries.findIndex(item => item.id === entry.id);
                entries.splice(index < 0 ? entries.length : index, 0, entry);
                await persist(entries); choose(entry); status('已保存');
            } else if (action === 'duplicate') {
                const entry = readEditor(); entry.id = crypto.randomUUID(); entry.title = uniqueTitle(entry.title, snapshot.entries);
                await persist([...snapshot.entries, entry]); choose(entry); status('副本已保存');
            } else if (action === 'delete-confirm') {
                await persist(snapshot.entries.filter(item => item.id !== selected)); choose(snapshot.entries[0]); status('存档已删除');
            }
        });
    });
    search.oninput = render;
    $('.pa-file').onchange = event => {
        const files = Array.from(event.target.files || []); event.target.value = '';
        if (files.length) void importFiles(files);
    };
    $('.pa-file').addEventListener('cancel', event => {
        // A file input's cancel event bubbles. It must not dismiss our dialog.
        event.stopPropagation();
        $('[data-action=import]').focus();
    });
    // Consume modal drops before they reach canvas/node import handlers, including
    // non-TXT files. The importer's document capture also explicitly skips us.
    for (const name of ['dragenter', 'dragover', 'dragleave', 'drop']) dialog.addEventListener(name, event => {
        event.preventDefault(); event.stopPropagation(); event.stopImmediatePropagation();
        if (name === 'dragenter' || name === 'dragover') { dialog.classList.add('pa-dragging'); if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'; }
        if (name === 'drop') { dialog.classList.remove('pa-dragging'); void importFiles(Array.from(event.dataTransfer?.files || [])); }
        if (name === 'dragleave' && !dialog.contains(event.relatedTarget)) dialog.classList.remove('pa-dragging');
    });
    dialog.addEventListener('keydown', event => event.stopPropagation());
    dialog.addEventListener('cancel', event => {
        if (event.target !== dialog) return;
        event.preventDefault(); close();
    });
    dialog.addEventListener('close', () => {
        if (active === dialog) active = null;
        dialog.remove();
        if (!active) previousFocus?.focus?.();
    }, { once: true });
    document.body.append(dialog); active = dialog; baseline = fingerprint(); dialog.showModal(); setBusy(true);
    try {
        const response = await fetchApi(ENDPOINT); const result = await response.json();
        if (!response.ok || !Array.isArray(result.entries)) throw Error(result.error || '无法读取提示词库');
        snapshot = result; choose(snapshot.entries[0]); status('');
    } catch (error) { status(error.message || String(error), true); }
    finally { setBusy(false); render(); }
}
