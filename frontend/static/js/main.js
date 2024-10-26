const API_URL = window.ENV?.API_URL || "http://localhost:8000";
console.log(API_URL)
// UI Elements
const errorBox = document.getElementById("errorMessage");
const video = document.getElementById("webcam");
const processedWebcam = document.getElementById("processedWebcam");
const canvas = document.createElement("canvas");
let stream = null, interval = null, ws = null;

// === Logging ===
function timestamp() {
    return new Date().toISOString();
}

function log(msg, level = "INFO") {
    console.log(`[${timestamp()}] [${level}] ${msg}`);
}

function showError(msg) {
    log(msg, "ERROR");
    errorBox.innerText = msg;
    errorBox.classList.remove("d-none");
}

function clearError() {
    errorBox.classList.add("d-none");
    errorBox.innerText = "";
}

// === Section Handling ===
function showSection(value) {
    clearError();
    stopWebcam();
    ["webcam", "image", "video"].forEach(id => {
        document.getElementById(`section-${id}`).classList.add("d-none");
    });
    document.getElementById(`section-${value}`).classList.remove("d-none");
    log(`Switched to section: ${value}`);
}

document.addEventListener("DOMContentLoaded", () => {
    const selected = document.getElementById("inputSelector").value;
    showSection(selected);
});

// === Webcam Streaming ===
function startWebcam() {
    navigator.mediaDevices.getUserMedia({video: true}).then(s => {
        stream = s;
        video.srcObject = s;

        const socketUrl = API_URL.replace(/^http/, "ws") + "/ws";
        ws = new WebSocket(socketUrl);
        ws.binaryType = "arraybuffer";

        let frameId = 0;
        let useEmoji = document.getElementById("emojiToggleWebcam")?.checked || false;

        // Send emoji config when WS is open
        ws.onopen = () => {
            log("WebSocket connection opened");
            const initialConfig = JSON.stringify({type: "config", emoji: useEmoji});
            ws.send(initialConfig);
            console.log("Initial config sent:", initialConfig);
        };

        ws.onerror = e => showError(`WebSocket error: ${e.message}`);
        ws.onclose = () => log("WebSocket closed");

        ws.onmessage = (event) => {
            const blob = new Blob([event.data], {type: "image/jpeg"});
            processedWebcam.src = URL.createObjectURL(blob);
        };

        // Attach toggle change listener
        const emojiToggle = document.getElementById("emojiToggleWebcam");
        if (emojiToggle) {
            emojiToggle.addEventListener("change", () => {
                useEmoji = emojiToggle.checked;
                if (ws.readyState === WebSocket.OPEN) {
                    const cfgMsg = JSON.stringify({type: "config", emoji: useEmoji});
                    ws.send(cfgMsg);
                    console.log("Updated config sent:", cfgMsg);
                }
            });
        }

        // Start sending frames
        interval = setInterval(() => {
            if (ws.readyState !== WebSocket.OPEN) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d").drawImage(video, 0, 0);

            canvas.toBlob(blob => {
                if (blob) {
                    blob.arrayBuffer().then(buf => {
                        ws.send(buf);
                        frameId++;
                    });
                }
            }, "image/jpeg", 0.7);
        }, 200);

    }).catch(err => showError("Webcam access failed: " + err.message));
}


function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
    clearInterval(interval);
    interval = null;
}


