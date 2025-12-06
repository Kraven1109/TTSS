import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// TTSS Audio Preview Extension
app.registerExtension({
    name: "TTSS.AudioPreview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TTSSPreviewAudio") {
            nodeType.prototype.onExecuted = function(message) {
                if (message.audio && message.audio.length >= 2) {
                    const audioName = message.audio[0];
                    const audioType = message.audio[1]; // "output" or "input"
                    
                    // Build URL using ComfyUI's view API
                    const params = new URLSearchParams({
                        filename: audioName,
                        type: audioType,
                    });
                    const audioUrl = api.apiURL('/view?' + params.toString());
                    
                    // Create audio widget if not exists
                    if (!this.audioWidget) {
                        const container = document.createElement("div");
                        container.className = "ttss_audio_preview";
                        container.style.width = "100%";
                        container.style.padding = "5px";
                        
                        const audioElement = document.createElement("audio");
                        audioElement.controls = true;
                        audioElement.style.width = "100%";
                        audioElement.addEventListener("loadedmetadata", () => {
                            fitHeight(this);
                        });
                        
                        container.appendChild(audioElement);
                        
                        this.audioWidget = this.addDOMWidget("audiopreview", "preview", container, {
                            serialize: false,
                            hideOnZoom: false,
                        });
                        this.audioWidget.audioEl = audioElement;
                    }
                    
                    // Update audio source
                    this.audioWidget.audioEl.src = audioUrl;
                    this.audioWidget.audioEl.hidden = false;
                    fitHeight(this);
                }
            };
        }
    }
});
