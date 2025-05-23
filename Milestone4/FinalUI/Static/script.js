function showUploadSection() {
    // Make the upload section visible
    document.getElementById("upload-section").style.display = "block";
    // Scrolls to the upload section smoothly
    document.getElementById("upload-section").scrollIntoView({ behavior: 'smooth' });
}
/// Function to handle file selection and show image preview
function showPreview() {
    const fileInput = document.getElementById('upload-input');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const analyzeContainer = document.getElementById('analyze-container');

    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;  // Show the image preview
            imagePreviewContainer.style.display = 'block';  // Show image preview container
            analyzeContainer.style.display = 'block';  // Show analyze button
        };
        reader.readAsDataURL(file);  // Read the file as a Data URL
    }
}

// Function to handle form submission with AJAX
function autoSubmit() {
    const formData = new FormData(document.getElementById('upload-form'));
    fetch('/', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // If there's a disease prediction, show it
        if (data.predicted_disease) {
            document.getElementById('disease-name').textContent = data.predicted_disease;
        }

        // Show the image from the backend if available
        if (data.img_base64) {
            document.getElementById('image-preview').src = `data:image/jpeg;base64,${data.img_base64}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to simulate image analysis with progress bar
function analyzeImage() {
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const analysisResult = document.getElementById('analysis-result');

    progressContainer.style.display = 'block';  // Show progress bar
    analysisResult.style.display = 'none';  // Hide result until analysis completes

    let progress = 0;
    const interval = setInterval(function () {
        if (progress < 100) {
            progress += 10;
            progressBar.style.width = progress + '%';
        } else {
            clearInterval(interval);
            showResult();  // Show result after progress completes
        }
    }, 500);
}

// Function to show the result after analysis
function showResult() {
    const analysisResult = document.getElementById('analysis-result');
    analysisResult.style.display = 'block';  // Show the result section
}

// Function to show cure and suggestions
function showCure() {
    document.getElementById('suggestionSection').style.display = 'block';
    document.querySelector('.buyBtn').style.display = 'inline-block';  // Show buy products button
}

// Function to navigate to AgriSupport page
function buyProduct() {
    window.location.href = "http://127.0.0.1:5000/agrisupport";
}
