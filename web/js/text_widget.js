import { app } from "../../../scripts/app.js";

function applyWidgetStyles(widget) {
    const el = widget.element;
    if (el) {
        el.readOnly = true;
        el.style.fontSize = "12px";
        el.style.whiteSpace = "pre-wrap";
		el.style.wordWrap = "break-word"; 
		el.style.height = "200px";
        el.style.overflowY = "auto";

	if (el.tagName === "TEXTAREA") {
            el.style.resize = "vertical";
        }
    } else {
        requestAnimationFrame(() => applyWidgetStyles(widget));
    }
}

app.registerExtension({
    name: "EasyLoraChecker.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EasyLoraCheckerToText") { // custom node name to applyed
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                // create textbox (readOnly)
                if (this.widgets && this.widgets.find(w => w.name === "output_text")) return;
				const widget = this.addWidget("TEXTAREA", "output_text", "", () => {}, { multiline: true });
                applyWidgetStyles(widget);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                // node update
                if (message && message.text) {
                    const w = this.widgets.find((w) => w.name === "output_text");
                    w.value = message.text[0]; // print to text
                }
            };
        }
    },
});