let predictionsChart = null;
let currentFile = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

analyzeBtn.addEventListener('click', () => {
    if (currentFile) {
        analyzeImage(currentFile);
    }
});

changeImageBtn.addEventListener('click', () => {
    resetForm();
});

newAnalysisBtn.addEventListener('click', () => {
    resetForm();
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Lütfen geçerli bir görüntü dosyası seçin.');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('Dosya boyutu 16MB\'dan büyük olamaz.');
        return;
    }
    
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        hideError();
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    hideError();
    loadingSection.style.display = 'block';
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Bir hata oluştu');
        }
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        loadingSection.style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('topClass').textContent = data.top_class;
    document.getElementById('confidenceBadge').textContent = `%${data.top_probability}`;
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = `${data.top_probability}%`;
    createChart(data.predictions);
    createPredictionsList(data.predictions);
    resultsSection.style.display = 'block';
}

function createChart(predictions) {
    const ctx = document.getElementById('predictionsChart').getContext('2d');
    if (predictionsChart) {
        predictionsChart.destroy();
    }
    const labels = predictions.map(p => p.class);
    const data = predictions.map(p => p.probability);
    const colors = predictions.map((p, i) => {
        if (i === 0) return 'rgba(102, 126, 234, 0.8)';
        return 'rgba(102, 126, 234, 0.4)';
    });
    predictionsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Olasılık (%)',
                data: data,
                backgroundColor: colors,
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createPredictionsList(predictions) {
    const listContainer = document.getElementById('predictionsList');
    listContainer.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const isTop = index === 0;
        if (isTop) {
            item.style.borderLeftColor = '#764ba2';
            item.style.background = '#f0f2ff';
        }
        
        item.innerHTML = `
            <span class="class-name">${pred.class}</span>
            <span class="probability">%${pred.probability.toFixed(2)}</span>
        `;
        
        listContainer.appendChild(item);
    });
}

function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}

function resetForm() {
    currentFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    hideError();
    
    if (predictionsChart) {
        predictionsChart.destroy();
        predictionsChart = null;
    }
}
