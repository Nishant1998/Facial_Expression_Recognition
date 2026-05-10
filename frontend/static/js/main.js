const API_URL = window.ENV?.API_URL || "http://localhost:8000";
console.log(API_URL)
// UI Elements
const errorBox = document.getElementById("errorMessage");
const video = document.getElementById("webcam");
const processedWebcam = document.getElementById("processedWebcam");
const canvas = document.createElement("canvas");
let stream = null, interval = null, ws = null;
let cameraSession = 0, animation = null, detections = [], lastResultAt = 0;
const emojiImages = new Map();
const objectUrls = new Map();

function replaceObjectUrl(key, blob) {
    if (objectUrls.has(key)) URL.revokeObjectURL(objectUrls.get(key));
    const url = URL.createObjectURL(blob);
    objectUrls.set(key, url);
    return url;
}

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

function showStatus(msg) {
    log(msg, "STATUS");
    const box = document.getElementById("statusMessage");
    if (box) {
        box.innerText = msg;
        box.classList.remove("d-none", "alert-danger");
        box.classList.add("alert-info");
    }
}

function clearStatus() {
    const box = document.getElementById("statusMessage");
    if (box) {
        box.classList.add("d-none");
        box.innerText = "";
    }
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
async function startWebcam() {
    stopWebcam();
    clearError();
    const session = cameraSession;
    const timing = document.getElementById('webcamTiming');
    try {
        const acquired = await navigator.mediaDevices.getUserMedia({video: true});
        if (session !== cameraSession) {
            acquired.getTracks().forEach(track => track.stop());
            return;
        }
        stream = acquired;
        video.srcObject = acquired;
        await video.play();
        if (session !== cameraSession) return;
        const socket = new WebSocket(API_URL.replace(/^http/, 'ws') + '/ws');
        ws = socket;
        let frameId = 0, waitingForReceipt = false, encoding = false;
        const sentAt = new Map();
        socket.onerror = () => {
            if (session === cameraSession) showError('Camera connection failed. Try restarting the camera.');
        };
        socket.onclose = () => {
            if (session === cameraSession) {
                stopWebcam();
                timing.textContent = 'Camera connection closed. Start the camera to reconnect.';
            }
        };
        socket.onmessage = event => {
            if (session !== cameraSession) return;
            const message = JSON.parse(event.data);
            if (message.type === 'received') {
                waitingForReceipt = false;
            } else if (message.type === 'result') {
                const capturedAt = sentAt.get(message.frame_id);
                const age = capturedAt === undefined ? Infinity : performance.now() - capturedAt;
                for (const id of sentAt.keys()) if (id <= message.frame_id) sentAt.delete(id);
                // A very old result should never be drawn over a current face.
                detections = age < 1500 ? message.detections : [];
                lastResultAt = performance.now();
                timing.textContent = `Result age: ${Math.round(age)} ms · processing: ${message.processing_ms} ms. Video stays live; labels update after processing.`;
            } else if (message.type === 'error') {
                detections = [];
                showError(message.message);
            }
        };
        interval = setInterval(async () => {
            if (session !== cameraSession || socket.readyState !== WebSocket.OPEN || waitingForReceipt || encoding || socket.bufferedAmount > 0 || !video.videoWidth) return;
            encoding = true;
            try {
                // Only this detection copy is reduced. Display uses the original video.
                const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
                canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
                canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                const capturedAt = performance.now();
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .85));
                if (!blob) return;
                const jpeg = new Uint8Array(await blob.arrayBuffer());
                if (session !== cameraSession || socket.readyState !== WebSocket.OPEN) return;
                const id = ++frameId;
                const packet = new Uint8Array(jpeg.length + 4);
                new DataView(packet.buffer).setUint32(0, id);
                packet.set(jpeg, 4);
                sentAt.set(id, capturedAt);
                // Keep memory bounded even if the server stops producing results.
                if (sentAt.size > 200) sentAt.delete(sentAt.keys().next().value);
                waitingForReceipt = true;
                socket.send(packet);
            } catch (error) {
                if (session === cameraSession) showError('Could not send camera frame: ' + error.message);
            } finally {
                encoding = false;
            }
        }, 100);

        function drawLiveVideo() {
            if (session !== cameraSession) return;
            if (video.readyState >= 2 && video.videoWidth) {
                if (processedWebcam.width !== video.videoWidth || processedWebcam.height !== video.videoHeight) {
                    processedWebcam.width = video.videoWidth;
                    processedWebcam.height = video.videoHeight;
                }
                const ctx = processedWebcam.getContext('2d');
                const width = processedWebcam.width, height = processedWebcam.height;
                ctx.drawImage(video, 0, 0, width, height);
                if (performance.now() - lastResultAt < 1500) {
                    for (const detection of detections) {
                        const [left, top, right, bottom] = detection.bbox;
                        const x = left * width, y = top * height;
                        const w = (right - left) * width, h = (bottom - top) * height;
                        if (document.getElementById('emojiToggleWebcam').checked) {
                            if (!emojiImages.has(detection.emoji)) {
                                const icon = new Image();
                                icon.src = `/emoji/${detection.emoji}.png`;
                                emojiImages.set(detection.emoji, icon);
                            }
                            const icon = emojiImages.get(detection.emoji);
                            if (icon.complete && icon.naturalWidth) {
                                ctx.drawImage(icon, x, y, w, h);
                                continue;
                            }
                        }
                        ctx.strokeStyle = '#ff3333';
                        ctx.fillStyle = '#ff3333';
                        ctx.lineWidth = 3;
                        ctx.font = `${Math.max(18, width / 35)}px sans-serif`;
                        ctx.strokeRect(x, y, w, h);
                        ctx.fillText(detection.label, x, Math.max(22, y - 6));
                    }
                }
            }
            animation = requestAnimationFrame(drawLiveVideo);
        }
        drawLiveVideo();
        timing.textContent = 'Camera is live. Waiting for the first expression result…';
    } catch (error) {
        if (session === cameraSession) {
            stopWebcam();
            showError('Webcam access failed: ' + error.message);
        }
    }
}


function stopWebcam() {
    cameraSession++;
    detections = [];
    lastResultAt = 0;
    if (animation !== null) cancelAnimationFrame(animation);
    animation = null;
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
    video.srcObject = null;
    processedWebcam.getContext('2d').clearRect(0, 0, processedWebcam.width, processedWebcam.height);
}

window.addEventListener('pagehide', () => {
    stopWebcam();
    for (const url of objectUrls.values()) URL.revokeObjectURL(url);
    objectUrls.clear();
});

// === Image Upload ===
function previewImage() {
    const file = document.getElementById("imageInput").files[0];
    if (file) {
        document.getElementById("originalImage").src = replaceObjectUrl('image-input', file);
        document.getElementById("processedBoxImage").innerHTML = '<span id="imageProcessingStatus">Waiting...</span>';
        log(`Image selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
    }
}

function sendImage() {
    const file = document.getElementById("imageInput").files[0];
    if (!file) return showError("Please select an image.");

    clearError();
    const statusText = document.getElementById("imageProcessingStatus");
    if (statusText) statusText.innerText = "Processing...";

    const useEmoji = document.getElementById("emojiToggleImage")?.checked || false;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("emoji", useEmoji);

    fetch(`${API_URL}/predict/image`, {
        method: "POST",
        body: formData
    })
        .then(async res => {
            if (!res.ok) {
                const errorJson = await res.json();
                throw new Error(errorJson.detail || errorJson.error || "Server error...");
            }
            return res.blob();
        })
        .then(blob => {
            const url = replaceObjectUrl('image-output', blob);
            document.getElementById("processedBoxImage").innerHTML = `<img src="${url}" />`;
            log("Processed image displayed");
        })
        .catch(err => {
            showError("Image processing failed: " + err.message);
        });
}

// === Video Upload ===
function previewVideo() {
    const file = document.getElementById("videoInput").files[0];
    if (file) {
        document.getElementById("originalVideo").src = replaceObjectUrl('video-input', file);
        document.getElementById("processedBoxVideo").innerHTML = "";
        log(`Video selected: ${file.name}, ${(file.size / 1024 / 1024).toFixed(2)} MB`);
    }
}

function sendVideo() {
    const file = document.getElementById("videoInput").files[0];
    if (!file) return showError("Please select a video.");

    clearError();
    showStatus("Uploading and processing video...");

    const useEmoji = document.getElementById("emojiToggleVideo")?.checked || false;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("emoji", useEmoji);

    fetch(`${API_URL}/predict/video`, {
        method: "POST",
        body: formData
    })
        .then(async res => {
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || error.error || 'Server error');
            }
            return res.blob();
        })
        .then(blob => {
            const url = replaceObjectUrl('video-output', blob);
            const container = document.getElementById("processedBoxVideo");
            container.innerHTML = `
                <video controls autoplay loop style="max-width: 100%;">
                    <source src="${url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>`;
            showStatus("Video successfully processed and displayed.");
            log("Processed video ready.");
        })
        .catch(err => {
            showError("Video processing failed: " + err.message);
        });
}
