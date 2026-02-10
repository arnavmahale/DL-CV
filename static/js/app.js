// Fresh or Rotten - Frontend JavaScript
// Handles camera access, image capture, and API communication

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');
const fileInput = document.getElementById('fileInput');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const resultsSection = document.getElementById('resultsSection');
const resultCard = document.getElementById('resultCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorMessage = document.getElementById('errorMessage');
const analyzeAgainBtn = document.getElementById('analyzeAgain');

// New elements for tab switching and upload
const methodTabs = document.querySelectorAll('.method-tab');
const cameraSection = document.getElementById('cameraSection');
const uploadSection = document.getElementById('uploadSection');
const uploadArea = document.getElementById('uploadArea');
const uploadPreview = document.getElementById('uploadPreview');
const previewImage = document.getElementById('previewImage');
const analyzeUploadBtn = document.getElementById('analyzeUpload');
const cancelUploadBtn = document.getElementById('cancelUpload');

// State
let stream = null;
let isCameraActive = false;
let currentImageBlob = null;
let continuousInferenceInterval = null;
let isAnalyzing = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkCameraSupport();
});

function setupEventListeners() {
    // Camera controls
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);

    // Upload controls
    fileInput.addEventListener('change', handleFileUpload);
    analyzeUploadBtn.addEventListener('click', analyzeCurrentImage);
    cancelUploadBtn.addEventListener('click', resetUpload);

    // Tab switching
    methodTabs.forEach(tab => {
        tab.addEventListener('click', () => switchMethod(tab.dataset.method));
    });

    // Drag and drop
    setupDragAndDrop();

    // Results
    analyzeAgainBtn.addEventListener('click', resetToInput);
}

// Tab Switching
function switchMethod(method) {
    // Update active tab
    methodTabs.forEach(tab => {
        if (tab.dataset.method === method) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Show appropriate section
    if (method === 'camera') {
        cameraSection.classList.add('active');
        uploadSection.classList.remove('active');
    } else {
        cameraSection.classList.remove('active');
        uploadSection.classList.add('active');
        // Stop camera when switching to upload
        if (isCameraActive) {
            stopCamera();
        }
    }

    // Hide results and errors
    resultsSection.style.display = 'none';
    hideError();
}

// Drag and Drop
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        });
    });

    uploadArea.addEventListener('drop', handleDrop);

    // Make upload area clickable (but not the button itself to avoid double-trigger)
    uploadArea.addEventListener('click', (e) => {
        // Don't trigger if clicking the label or its children
        if (e.target.closest('label')) return;
        fileInput.click();
    });
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function checkCameraSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Camera not supported on this device/browser. Please use file upload.');
        startCameraBtn.disabled = true;
    }
}

async function startCamera() {
    try {
        // Request camera access with mobile-optimized constraints
        const constraints = {
            video: {
                facingMode: 'environment', // Use back camera on mobile
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        // Update UI
        cameraPlaceholder.style.display = 'none';
        isCameraActive = true;
        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;

        hideError();

        // Start continuous inference
        startContinuousInference();
    } catch (error) {
        console.error('Camera error:', error);
        if (error.name === 'NotAllowedError') {
            showError('Camera permission denied. Please allow camera access and try again.');
        } else if (error.name === 'NotFoundError') {
            showError('No camera found on this device. Please use file upload.');
        } else {
            showError('Failed to access camera: ' + error.message);
        }
    }
}

function stopCamera() {
    // Stop continuous inference
    stopContinuousInference();

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }

    // Update UI
    cameraPlaceholder.style.display = 'flex';
    isCameraActive = false;
    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;

    // Hide results
    resultsSection.style.display = 'none';
}

// Continuous Inference Functions
function startContinuousInference() {
    // Initial analysis after a short delay to let camera stabilize
    setTimeout(() => {
        if (isCameraActive) {
            captureAndAnalyze();
        }
    }, 1000);

    // Set up continuous inference every 2.5 seconds
    continuousInferenceInterval = setInterval(() => {
        if (isCameraActive && !isAnalyzing) {
            captureAndAnalyze();
        }
    }, 2500);
}

function stopContinuousInference() {
    if (continuousInferenceInterval) {
        clearInterval(continuousInferenceInterval);
        continuousInferenceInterval = null;
    }
    isAnalyzing = false;
}

function captureAndAnalyze() {
    if (!isCameraActive || isAnalyzing) {
        return;
    }

    // Capture frame from video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
        if (blob && isCameraActive) {
            await analyzeImageContinuous(blob);
        }
    }, 'image/jpeg', 0.85);
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    handleFile(file);
    event.target.value = ''; // Reset file input
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image (JPEG, PNG, WebP, BMP).');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }

    // Store the file
    currentImageBlob = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        uploadPreview.style.display = 'block';
        hideError();
    };
    reader.onerror = () => {
        showError('Failed to read the image file. Please try again.');
        currentImageBlob = null;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    uploadArea.style.display = 'block';
    uploadPreview.style.display = 'none';
    previewImage.src = '';
    currentImageBlob = null;
    fileInput.value = '';
}

async function analyzeCurrentImage() {
    if (!currentImageBlob) {
        showError('No image selected');
        return;
    }
    await analyzeImage(currentImageBlob);
}

async function analyzeImage(imageBlob) {
    // Show loading
    showLoading();
    hideError();
    resultsSection.style.display = 'none';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', imageBlob, 'image.jpg');

        // Send to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Analysis error:', error);
        showError('Analysis failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function analyzeImageContinuous(imageBlob) {
    if (isAnalyzing) return;

    isAnalyzing = true;
    hideError();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', imageBlob, 'image.jpg');

        // Send to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Display results (without hiding the camera)
        displayResultsContinuous(data);

    } catch (error) {
        console.error('Analysis error:', error);
        // Don't show error in continuous mode to avoid interruption
    } finally {
        isAnalyzing = false;
    }
}

function displayResults(data) {
    // Update prediction label and icon
    const predictionLabel = document.getElementById('predictionLabel');
    const resultIcon = document.getElementById('resultIcon');

    predictionLabel.textContent = data.prediction;

    // Set icon and color based on prediction
    if (data.prediction === 'Fresh') {
        resultIcon.textContent = '✅';
        resultCard.classList.add('fresh');
        resultCard.classList.remove('rotten');
    } else {
        resultIcon.textContent = '❌';
        resultCard.classList.add('rotten');
        resultCard.classList.remove('fresh');
    }

    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');

    confidenceValue.textContent = data.confidence.toFixed(1) + '%';
    confidenceFill.style.width = data.confidence + '%';

    // Update probabilities
    document.getElementById('freshProb').textContent =
        data.probabilities.Fresh.toFixed(1) + '%';
    document.getElementById('rottenProb').textContent =
        data.probabilities.Rotten.toFixed(1) + '%';

    // Update inference time
    document.getElementById('inferenceTime').textContent =
        data.inference_time.toFixed(2);

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayResultsContinuous(data) {
    // Same as displayResults but keeps camera visible
    const predictionLabel = document.getElementById('predictionLabel');
    const resultIcon = document.getElementById('resultIcon');

    predictionLabel.textContent = data.prediction;

    // Set icon and color based on prediction
    if (data.prediction === 'Fresh') {
        resultIcon.textContent = '✅';
        resultCard.classList.add('fresh');
        resultCard.classList.remove('rotten');
    } else {
        resultIcon.textContent = '❌';
        resultCard.classList.add('rotten');
        resultCard.classList.remove('fresh');
    }

    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');

    confidenceValue.textContent = data.confidence.toFixed(1) + '%';
    confidenceFill.style.width = data.confidence + '%';

    // Update probabilities
    document.getElementById('freshProb').textContent =
        data.probabilities.Fresh.toFixed(1) + '%';
    document.getElementById('rottenProb').textContent =
        data.probabilities.Rotten.toFixed(1) + '%';

    // Update inference time
    document.getElementById('inferenceTime').textContent =
        data.inference_time.toFixed(2);

    // Show results without scrolling
    resultsSection.style.display = 'block';
}

function resetToInput() {
    resultsSection.style.display = 'none';
    hideError();

    // Reset upload preview if in upload mode
    if (uploadSection.classList.contains('active')) {
        resetUpload();
    }

    // Scroll back to input section
    const activeSection = document.querySelector('.input-section.active');
    if (activeSection) {
        activeSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// Handle page visibility changes (pause camera when tab is hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isCameraActive) {
        // Pause video when tab is hidden to save resources
        video.pause();
    } else if (!document.hidden && isCameraActive) {
        video.play();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});

// Service worker registration (for PWA support - optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable PWA features
        // navigator.serviceWorker.register('/sw.js');
    });
}
