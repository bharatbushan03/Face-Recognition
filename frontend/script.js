document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    const btnRecognize = document.getElementById('btn-recognize');
    const btnRegister = document.getElementById('btn-register');
    const usernameInput = document.getElementById('username');
    const resultDiv = document.getElementById('recognition-result');
    const scanOverlay = document.querySelector('.scan-overlay');
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
                    resultDiv.className = 'success';
                    resultDiv.textContent = `Match: ${data.user.name} (Conf: ${Math.round(data.confidence * 100)}%)`;
                    showToast(`Welcome back, ${data.user.name}!`, "success");
                } else {
                    resultDiv.className = 'error';
                    resultDiv.textContent = 'Unknown Face';
                    showToast("Face not recognized in database.", "error");
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

    // Start
    initCamera();
});
