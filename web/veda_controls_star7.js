import { app } from "/scripts/app.js";

const TITLE = "MiniMax H3 VEDA 稀疏注意力 - Star7";
// Preserve the original six positional values in saved workflows.
const FIELDS = ["predictor", "generated_sparsity", "reference_sparsity",
    "full_attention_layers", "full_attention_steps", "verbose", "enabled"];
const LABELS = { predictor: "VEDA 预测模型", generated_sparsity: "生成区域稀疏度",
    reference_sparsity: "参考区域稀疏度", full_attention_layers: "完整注意力层",
    full_attention_steps: "完整注意力步", verbose: "详细日志", enabled: "启用 VEDA" };
const EN_LABELS = { predictor: "VEDA predictor", generated_sparsity: "Generated sparsity",
    reference_sparsity: "Reference sparsity", full_attention_layers: "Full-attention layers",
    full_attention_steps: "Full-attention steps", verbose: "Detailed logs", enabled: "Enable VEDA" };
function chinese() {
    return String(app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language ?? "zh-CN").toLowerCase().startsWith("zh");
}
function title() { return chinese() ? TITLE : "MiniMax H3 VEDA Sparse Attention - Star7"; }

function order(node, names) {
    const priority = new Map(names.map((name, index) => [name, index]));
    node.widgets?.sort((a, b) => (priority.get(a.name) ?? 99) - (priority.get(b.name) ?? 99));
}

function present(node) {
    if (!node.title || ["Star7VedaSparseAttention", "Star7 VEDA 稀疏注意力 - Star7",
        "Star7 VEDA 稀疏注意力 · MiniMax H3", TITLE,
        "MiniMax H3 VEDA Sparse Attention - Star7"].includes(node.title)) {
        node.title = title();
    }
    order(node, ["enabled", ...FIELDS.filter(name => name !== "enabled")]);
    for (const widget of node.widgets ?? []) {
        const labels = chinese() ? LABELS : EN_LABELS;
        if (labels[widget.name]) widget.label = labels[widget.name];
    }
    for (const socket of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
        if (socket.name === "model") socket.label = chinese() ? "模型" : "Model";
    }
}

app.registerExtension({
    name: "Star7.VedaControls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Star7VedaSparseAttention") return;
        // Update the definition used by the search menu as well as the canvas title.
        nodeData.display_name = title();
        nodeType.title = title();
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            present(this);
            return result;
        };
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (configuration) {
            const positional = configuration.widgets_values ?? [];
            const named = configuration.widgets_values_named ?? {};
            const values = FIELDS.map((name, index) => named[name] ?? positional[index]
                ?? (name === "enabled" ? true : this.widgets?.find(w => w.name === name)?.value));
            order(this, FIELDS);
            try {
                const result = configure?.apply(this, arguments);
                FIELDS.forEach((name, index) => {
                    const widget = this.widgets?.find(w => w.name === name);
                    if (widget) widget.value = values[index];
                });
                return result;
            } finally {
                present(this);
            }
        };
        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const result = serialize?.apply(this, arguments) ?? {};
            const named = Object.fromEntries(FIELDS.map(name => [name,
                this.widgets?.find(w => w.name === name)?.value]));
            result.widgets_values_named = named;
            result.widgets_values = FIELDS.map(name => named[name]);
            return result;
        };
    },
});
