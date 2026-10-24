import { app } from '/scripts/app.js';
const NAME='MiniMaxH3UnifiedAccelerationStar7';
const TITLE='MiniMax H3 Spark / VPack 加速 - Star7';
const fields=['enabled','mode','topk_ratio','warmup_ratio','vpack_ratio'];
const labels={enabled:'启用加速',mode:'加速模式',topk_ratio:'Spark Top-K 比例',warmup_ratio:'Spark 预热比例',vpack_ratio:'VPack 压缩比例'};
const recipes={balanced:[.2,.2,'2/5'],fidelity:[.3,.2,'1/5'],speed:[.1,0,'2/5']};
function migrate(values){
    if(!Array.isArray(values))return null;
    if(typeof values[0]==='boolean')return values.slice(0,5);
    let mode=values[0],topk=values[1],warmup=values[2],ratio=values[3];
    if(values.length>=12 && typeof values[1]==='string'){
        const recipe=recipes[values[1]];
        topk=recipe?.[0]??values[5];warmup=recipe?.[1]??values[6];ratio=recipe?.[2]??values[3];
    }
    return [mode!=='off',mode==='off'?'spark':mode==='auto'?'vpack-only':mode,topk,warmup,ratio];
}
app.registerExtension({
    name:'Star7.H3.SparkVPack',
    beforeRegisterNodeDef(Node,data){
        if(data.name!==NAME)return;
        const created=Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated=function(){
            created?.apply(this,arguments);
            const enabled=this.widgets?.find(w=>w.name==='enabled');
            if(enabled)this.widgets=[enabled,...this.widgets.filter(w=>w!==enabled)];
            for(const widget of this.widgets||[])if(labels[widget.name])widget.label=labels[widget.name];
            if(this.computeSize&&this.setSize)this.setSize(this.computeSize());
        };
        const configured=Node.prototype.onConfigure;
        Node.prototype.onConfigure=function(data){
            const values=migrate(data.widgets_values);
            if(values)data.widgets_values=values;
            configured?.apply(this,arguments);
            if(values)for(const [index,name] of fields.entries()){
                const widget=this.widgets?.find(w=>w.name===name);if(widget)widget.value=values[index];
            }
            if(!data.title||/统一加速|加速配件/.test(data.title))this.title=TITLE;
            if(this.computeSize&&this.setSize)this.setSize(this.computeSize());
        };
    },
});
