import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3VDNStar7";

app.registerExtension({
    name: "star7.h3.vdn",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = previous?.apply(this, arguments);
            this.title = "MiniMax H3 VDN 加速 - Star7";
            const labels = {
                model: "模型",
                vdn_checkpoint: "VDN 模型",
                inference_mode: "运行模式",
            };
            for (const widget of this.widgets || []) {
                if (labels[widget.name]) widget.label = labels[widget.name];
            }
            return result;
        };
    },
});
