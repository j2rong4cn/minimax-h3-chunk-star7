import { app } from "/scripts/app.js";

const NODE_NAME = "Star7DLSSNeuralEnhance";
const CUSTOM = "自定义参数";
const SERIAL_NAMES = ["风格预设", "目标像素 (MP)", "NR 强度", "局部结构", "局部色调", "皮肤结构", "时序稳定", "自动蒙版", "模型"];
const PARAMS = ["NR 强度", "局部结构", "局部色调", "皮肤结构", "时序稳定", "自动蒙版"];
const REALISTIC = {
    "NR 强度": 0.90,
    "局部结构": 0.70,
    "局部色调": 0.75,
    "皮肤结构": -1.0,
    "时序稳定": 0.55,
    "自动蒙版": true,
};
const PRESETS = {
    "真实风格": REALISTIC,
    "真实人像优化": { "NR 强度": 0.85, "局部结构": 0.60, "局部色调": 0.70, "皮肤结构": 0.00, "时序稳定": 0.60, "自动蒙版": true },
    "3D 动漫风格": { "NR 强度": 1.20, "局部结构": 1.25, "局部色调": 0.95, "皮肤结构": -1.0, "时序稳定": 0.50, "自动蒙版": false },
    "2D 动漫风格": { "NR 强度": 1.05, "局部结构": 1.00, "局部色调": 0.70, "皮肤结构": -1.0, "时序稳定": 0.45, "自动蒙版": false },
};

function widgetMap(node) {
    return Object.fromEntries((node.widgets || []).map((widget) => [widget.name, widget]));
}

function snapshot(widgets) {
    return Object.fromEntries(PARAMS.map((name) => [name, widgets[name]?.value]));
}

function applyValues(node, widgets, values) {
    node._star7ApplyingPreset = true;
    try {
        for (const name of PARAMS) {
            if (!widgets[name] || !(name in values)) continue;
            widgets[name].value = values[name];
            widgets[name].callback?.(values[name]);
        }
    } finally {
        node._star7ApplyingPreset = false;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "star7.dlss-neural-enhance",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalCreated?.apply(this, args);
            const widgets = widgetMap(this);
            const model = widgets["模型"];
            if (model) {
                this.widgets.splice(this.widgets.indexOf(model), 1);
                this.widgets.unshift(model);
            }
            const preset = widgets["风格预设"];
            if (!preset) return result;
            if (Array.isArray(preset.options?.values)) {
                preset.options.values = preset.options.values.filter((value) => value !== "真实风格加强");
            }

            this.properties ||= {};
            this.properties.star7_dlss_custom ||= { ...REALISTIC };
            this._star7Preset = preset.value;

            const originalPresetCallback = preset.callback;
            preset.callback = (value, ...callbackArgs) => {
                const previous = this._star7Preset;
                if (previous === CUSTOM && !this._star7ApplyingPreset) {
                    this.properties.star7_dlss_custom = snapshot(widgets);
                }
                originalPresetCallback?.call(preset, value, ...callbackArgs);
                this._star7Preset = value;
                if (value === CUSTOM) {
                    applyValues(this, widgets, this.properties.star7_dlss_custom || REALISTIC);
                } else if (PRESETS[value]) {
                    applyValues(this, widgets, PRESETS[value]);
                }
            };

            for (const name of PARAMS) {
                const widget = widgets[name];
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = (value, ...callbackArgs) => {
                    originalCallback?.call(widget, value, ...callbackArgs);
                    if (this._star7ApplyingPreset) return;
                    if (preset.value !== CUSTOM) {
                        preset.value = CUSTOM;
                        this._star7Preset = CUSTOM;
                    }
                    this.properties.star7_dlss_custom = snapshot(widgets);
                };
            }

            const resetWidget = this.addWidget("button", "重置参数", "重置参数", () => {
                const values = preset.value === CUSTOM
                    ? { ...REALISTIC }
                    : { ...(PRESETS[preset.value] || REALISTIC) };
                if (preset.value === CUSTOM) {
                    this.properties.star7_dlss_custom = { ...values };
                }
                // Assign directly while the guard is active. This avoids a quality
                // widget callback turning the node back into Custom mid-reset.
                this._star7ApplyingPreset = true;
                try {
                    for (const name of PARAMS) {
                        if (widgets[name] && name in values) widgets[name].value = values[name];
                    }
                } finally {
                    this._star7ApplyingPreset = false;
                }
                this.graph?.change?.();
                this.graph?.setDirtyCanvas?.(true, true);
                this.setDirtyCanvas?.(true, true);
            }, { serialize: false });
            resetWidget.serializeValue = () => undefined;

            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            originalSerialize?.call(this, info);
            const widgets = widgetMap(this);
            // UI order differs from the append-only backend schema.
            if (widgets["模型"]) info.widgets_values = SERIAL_NAMES.map(name => widgets[name]?.value);
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            const widgets = widgetMap(this);
            const saved = args[0]?.widgets_values;
            if (widgets["模型"] && Array.isArray(saved)) {
                SERIAL_NAMES.forEach((name, index) => {
                    if (widgets[name] && saved[index] != null) widgets[name].value = saved[index];
                });
                if (saved[8] == null) widgets["模型"].value = "nvngx_dlssnr.dll";
            }
            if (widgets["模型"] && this.computeSize && this.setSize) {
                this.setSize([this.size[0], Math.max(this.size[1], this.computeSize([this.size[0], 0])[1])]);
            }
            const preset = widgets["风格预设"];
            if (preset?.value === "真实风格加强") {
                preset.value = "真实风格";
                this._star7Preset = "真实风格";
                applyValues(this, widgets, REALISTIC);
            }
            return result;
        };
    },
});
