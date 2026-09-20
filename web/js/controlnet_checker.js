import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "EasyControlNetChecker.UnifiedUI",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "EasyControlNetChecker") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      this._widgetInitialized = false;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);

      const text = message?.text?.[0];
      const fontSize = message?.font_size?.[0];
      const useWidget2 = message?.widget?.[0];   //

      if (!this._widgetInitialized) {
        if (useWidget2) {
          // --- 2.0: Text Area Widget ---
          const widget = this.addWidget("TEXTAREA", "output_text", "", () => {}, { multiline: true });
          const applyStyles = () => {
            const el = widget.element;
            if (el) {
              el.readOnly = true;
              el.style.fontSize = "12px";
              el.style.whiteSpace = "pre-wrap";
              el.style.wordWrap = "break-word";
			  el.style.height = "200px";
              el.style.overflowY = "auto";
              if (el.tagName === "TEXTAREA") el.style.resize = "vertical";
            } else {
              requestAnimationFrame(applyStyles);
            }
          };
          applyStyles();
          this._textareaWidget = widget;
        } else {
          // --- 1.0 (Default): DOM widget ---
          const container = document.createElement("div");
          container.style.padding = "4px";
          container.style.fontFamily = "sans-serif";
          container.style.fontSize = "14px";
          container.style.color = "#ffd700";
          container.style.whiteSpace = "pre-line";
          container.style.backgroundColor = "#3a3a3a";
          container.innerText = "No result";
          container.style.height = "200px";
          container.style.overflowY = "auto";
          container.style.userSelect = "text";
          container.style.webkitUserSelect = "text";
          container.style.cursor = "text"; 

          this.addDOMWidget("stats_display", "display", container, { serialize: false });
          this._statsContainer = container;
        }
        this._widgetInitialized = true;
      }

      if (this._textareaWidget) {
        if (text) this._textareaWidget.value = text;
        if (fontSize && this._textareaWidget.element) this._textareaWidget.element.style.fontSize = String(fontSize) + "px";
      } else if (this._statsContainer) {
        if (text) this._statsContainer.innerText = text;
        if (fontSize) this._statsContainer.style.fontSize = fontSize + "px";
      }
    };
  },
});