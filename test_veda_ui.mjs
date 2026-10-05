import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

let extension;
const code = fs.readFileSync(new URL('./web/veda_controls_star7.js', import.meta.url), 'utf8')
    .replace('import { app } from "/scripts/app.js";', '');
vm.runInNewContext(code, { app: { registerExtension: value => extension = value } });
const fields = ['predictor','generated_sparsity','reference_sparsity',
    'full_attention_layers','full_attention_steps','verbose','enabled'];
class Node {
    constructor() {
        this.widgets = fields.map(name => ({name,value: name === 'enabled' ? true : null}));
        this.onNodeCreated();
    }
    configure(data) { data.widgets_values.forEach((v,i) => this.widgets[i].value = v); }
    serialize() { return {widgets_values:this.widgets.map(w => w.value)}; }
}
const definition = {name:'Star7VedaSparseAttention',display_name:'Star7 VEDA 稀疏注意力 · MiniMax H3'};
extension.beforeRegisterNodeDef(Node, definition);
assert.equal(definition.display_name, 'MiniMax H3 VEDA 稀疏注意力 - Star7');
assert.equal(Node.title, definition.display_name);
const node = new Node();
const original = ['predictor.safetensors','90%','80%','0,49','0',true];
for (let i=0; i<2; i++) {
    node.configure({widgets_values:original});
    assert.equal(node.widgets[0].name, 'enabled');
    assert.equal(node.widgets[0].value, true);
    assert.deepEqual(Array.from(node.serialize().widgets_values), [...original,true]);
}
node.widgets[0].value = false;
const saved = node.serialize();
const reopened = new Node();
reopened.configure(saved);
assert.equal(reopened.widgets[0].value, false);
assert.deepEqual(Array.from(reopened.serialize().widgets_values), [...original,false]);
assert.equal(reopened.widgets.find(w => w.name === 'predictor').label, 'VEDA 预测模型');
vm.runInNewContext(code, { app: { ui: {settings: {getSettingValue: () => 'en'}}, registerExtension: value => extension = value } });
class EnglishNode { constructor() { this.widgets = fields.map(name => ({name})); this.onNodeCreated(); } }
const englishDef = {name:'Star7VedaSparseAttention'};
extension.beforeRegisterNodeDef(EnglishNode, englishDef);
const englishNode = new EnglishNode();
assert.equal(englishDef.display_name, 'MiniMax H3 VEDA Sparse Attention - Star7');
assert.equal(englishNode.widgets[0].label, 'Enable VEDA');
assert.equal(englishNode.widgets.find(w => w.name === 'predictor').label, 'VEDA predictor');
console.log('Top switch / legacy six fields / disabled round trip / bilingual labels: PASS');
