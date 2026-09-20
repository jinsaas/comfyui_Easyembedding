import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "EasyVAECheckerWidget",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "EasyVAEChecker") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

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

      this.addDOMWidget("stats_display", "display", container, {
        serialize: false,
      });

      this._statsContainer = container;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.stats?.[0];
      const fontSize = message?.font_size?.[0];
      if (this._statsContainer && text) {
        this._statsContainer.innerText = text;
		if (fontSize) {
 		   this._statsContainer.style.fontSize = fontSize + "px";
		}
      }
    };
  },
});