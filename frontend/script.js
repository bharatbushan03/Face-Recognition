document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    const btnRecognize = document.getElementById('btn-recognize');
    const btnRegister = document.getElementById('btn-register');
    const usernameInput = document.getElementById('username');
    const resultDiv = document.getElementById('recognition-result');
    const scanOverlay = document.querySelector('.scan-overlay');
    const hudCanvas = document.getElementById('hudCanvas');
    const btnToggleRoster = document.getElementById('btn-toggle-roster');
    const rosterList = document.getElementById('roster-list');
    const constaints = { video: { facingMode: "user" } };

    const API_BASE = "http://localhost:8000/api/face";

    // Initialize Camera
    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constaints);
            video.srcObject = stream;
        } catch (err) {
            console.error("Camera error:", err);
            showToast("Failed to access camera. Please check permissions.", "error");
            btnRecognize.disabled = true;
            btnRegister.disabled = true;
        }
    }

    // Voice Synthesis Helper
    function speakName(name, isSmiling) {
        if ('speechSynthesis' in window) {
            const greeting = isSmiling ? `Access Granted. You look happy today, ${name}!` : `Access Granted. Welcome back, ${name}.`;
            const msg = new SpeechSynthesisUtterance(greeting);
            msg.rate = 1.0;
            msg.pitch = 1.0;
            window.speechSynthesis.speak(msg);
        }
    }

    // HUD Drawing Logic
    function drawHUD(box, name, confidence) {
        hudCanvas.width = video.videoWidth;
        hudCanvas.height = video.videoHeight;
        const ctx = hudCanvas.getContext('2d');
        ctx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

        if (!box) return;

        // Box is [top, right, bottom, left]
        // Since the image was captured flipped, we need to map it back without flipping the canvas context
        // so that text isn't backwards.
        const originalTop = box[0];
        const originalRight = box[1];
        const originalBottom = box[2];
        const originalLeft = box[3];

        const targetLeft = hudCanvas.width - originalRight;
        const targetTop = originalTop;
        const targetWidth = originalRight - originalLeft;
        const targetHeight = originalBottom - originalTop;

        // Draw animated glowing box
        ctx.strokeStyle = '#00f3ff';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#00f3ff';
        ctx.shadowBlur = 15;
        
        // Draw corners
        ctx.strokeRect(targetLeft, targetTop, targetWidth, targetHeight);

        // Draw label background
        ctx.fillStyle = 'rgba(0, 243, 255, 0.2)';
        ctx.fillRect(targetLeft, targetTop - 30, targetWidth, 30);

        // Draw text
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Inter, sans-serif';
        ctx.fillText(`${name} (${Math.round(confidence * 100)}%)`, targetLeft + 5, targetTop - 10);
        
        // Clear HUD after 4 seconds
        setTimeout(() => {
            ctx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        }, 4000);
    }

    // Capture Image from Video Feed
    function captureImageBase64() {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame onto the canvas
        // Need to flip the canvas horizontally since CSS mirrors the video display
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get base64 representation
        return canvas.toDataURL('image/jpeg', 0.8);
    }

    // Helper: Set Button Loading State
    function setBtnLoading(btn, isLoading) {
        const textSpan = btn.querySelector('.btn-text');
        const loader = btn.querySelector('.loader');
        
        if (isLoading) {
            btn.disabled = true;
            textSpan.style.display = 'none';
            loader.style.display = 'block';
            scanOverlay.classList.add('active');
        } else {
            btn.disabled = false;
            textSpan.style.display = 'block';
            loader.style.display = 'none';
            scanOverlay.classList.remove('active');
        }
    }

    // Recognize Face
    btnRecognize.addEventListener('click', async () => {
        resultDiv.style.display = 'none';
        setBtnLoading(btnRecognize, true);

        const base64Image = captureImageBase64();
        const formData = new FormData();
        formData.append("image_base64", base64Image);

        try {
            const response = await fetch(`${API_BASE}/recognize`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                if (data.match_found) {
                    const smileText = data.is_smiling ? " 😊" : "";
                    resultDiv.className = 'success';
                    resultDiv.textContent = `Match: ${data.user.name}${smileText} (Conf: ${Math.round(data.confidence * 100)}%)`;
                    showToast(`Welcome back, ${data.user.name}!`, "success");
                    speakName(data.user.name, data.is_smiling);
                    drawHUD(data.box, data.user.name, data.confidence);
                } else {
                    resultDiv.className = 'error';
                    resultDiv.textContent = 'Unknown Face';
                    showToast("Face not recognized in database.", "error");
                    if (data.box) {
                        drawHUD(data.box, "Unknown Target", data.confidence);
                        if ('speechSynthesis' in window) {
                            window.speechSynthesis.speak(new SpeechSynthesisUtterance("Intruder detected. Access Denied."));
                        }
                    }
                }
            } else {
                handleApiError(data);
            }
        } catch (err) {
            showToast("Network error occurred.", "error");
            console.error(err);
        } finally {
            setBtnLoading(btnRecognize, false);
        }
    });

    // Register User
    btnRegister.addEventListener('click', async () => {
        const username = usernameInput.value.trim();
        if (!username) {
            showToast("Please enter a name first.", "error");
            usernameInput.focus();
            return;
        }

        setBtnLoading(btnRegister, true);

        const base64Image = captureImageBase64();
        const formData = new FormData();
        formData.append("image_base64", base64Image);
        formData.append("name", username);

        try {
            const response = await fetch(`${API_BASE}/register`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                showToast(`Successfully registered ${data.user.name}!`, "success");
                usernameInput.value = '';
            } else {
                handleApiError(data);
            }
        } catch (err) {
            showToast("Network error occurred.", "error");
            console.error(err);
        } finally {
            setBtnLoading(btnRegister, false);
        }
    });

    // Common API error handler
    function handleApiError(data) {
        if (data && data.message) {
            showToast(data.message, "error");
        } else {
            showToast("An unexpected error occurred.", "error");
        }
    }

    // Toast functionality
    function showToast(message, type = "success") {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Trigger reflow for transition
        setTimeout(() => toast.classList.add('show'), 10);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // Roster logic
    btnToggleRoster.addEventListener('click', async () => {
        if (rosterList.style.display === 'none') {
            try {
                const res = await fetch(`${API_BASE}/users`);
                const users = await res.json();
                
                rosterList.innerHTML = '';
                if (users.length === 0) {
                    rosterList.innerHTML = '<div class="roster-item">No users registered yet</div>';
                } else {
                    users.forEach(user => {
                        const date = new Date(user.created_at).toLocaleDateString();
                        rosterList.innerHTML += `
                            <div class="roster-item">
                                <span>${user.name}</span>
                                <span class="roster-badge">Active</span>
                            </div>
                        `;
                    });
                }
                rosterList.style.display = 'flex';
                btnToggleRoster.textContent = 'Hide Registered Users';
            } catch (e) {
                showToast("Failed to fetch users", "error");
            }
        } else {
            rosterList.style.display = 'none';
            btnToggleRoster.textContent = 'View Registered Users';
        }
    });

    // Start
    initCamera();
});
