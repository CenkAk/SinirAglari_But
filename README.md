# Göz Hastalığı Tespit Sistemi

Fundus kamera görüntülerinden göz hastalıklarını tespit eden yapay zeka destekli karar destek sistemi. Bu sistem, EfficientNet tabanlı derin öğrenme modeli kullanarak 10 farklı göz hastalığını ve sağlıklı göz görüntülerini sınıflandırmaktadır.

##  İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Eğitimi](#model-eğitimi)
- [Proje Yapısı](#proje-yapısı)
- [Teknolojiler](#teknolojiler)

##  Özellikler

- **10 Farklı Hastalık Sınıflandırması:**
  - Retinitis Pigmentosa
  - Retina Dekolmanı
  - Pterjium
  - Miyopi
  - Maküler Skar
  - Glokom
  - Disk Ödemesi
  - Diyabetik Retinopati
  - Santral Seröz Korioretinopati
  - Sağlıklı Göz

- **Modern Web Arayüzü:**
  - Drag & drop görüntü yükleme
  - Gerçek zamanlı analiz
  - Detaylı sonuç görselleştirmesi
  - Responsive tasarım

- **Gelişmiş Model Mimarisi:**
  - EfficientNet-B3 transfer learning
  - İki aşamalı eğitim stratejisi
  - Veri artırma teknikleri
  - Sınıf dengesizliği için ağırlıklandırma

## Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- CUDA destekli GPU (önerilir, opsiyonel)

### Adımlar

1. **Virtual environment oluşturun (önerilir):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# veya
source venv/bin/activate  # Linux/Mac
```

2. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Veri setini hazırlayın:**
   - `Eye Disease Image Dataset` klasöründe zip dosyalarınızın olduğundan emin olun
   - Veya veri setini `data/original/` veya `data/augmented/` klasörlerine yerleştirin

##  Kullanım

### Model Eğitimi

Modeli eğitmek için:

```bash
python train.py
```

Eğitim sırasında:
- Veri seti otomatik olarak yüklenecek ve ön işlenecek
- Model iki aşamada eğitilecek (frozen + fine-tuning)
- En iyi model `models/best_model.h5` olarak kaydedilecek
- Değerlendirme metrikleri ve görselleri `models/` klasörüne kaydedilecek

### Web Uygulamasını Çalıştırma

1. **Model eğitilmiş olmalı** (`models/best_model.h5` dosyası mevcut olmalı)

2. **Flask uygulamasını başlatın:**
```bash
python app.py
```

3. **Tarayıcıda açın:**
```
http://localhost:5000
```

4. **Görüntü yükleyin ve analiz edin:**
   - Görüntüyü sürükleyip bırakın veya tıklayarak seçin
   - "Analiz Et" butonuna tıklayın
   - Sonuçları görüntüleyin

##  Model Eğitimi Detayları

### Eğitim Parametreleri

- **Görüntü Boyutu:** 380x380 piksel
- **Batch Size:** 32
- **Optimizer:** Adam (lr=0.0001, fine-tuning: 0.00001)
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 
  - Frozen stage: 10 epoch
  - Fine-tuning: 50 epoch (early stopping ile)

### Veri Artırma

- Random rotation (±15°)
- Width/Height shift (±10%)
- Shear transformation
- Zoom (0.9-1.1x)
- Horizontal/Vertical flip
- Brightness adjustment (±20%)

### Model Mimarisi

```
Input (380x380x3)
  ↓
EfficientNet-B3 (pre-trained, ImageNet)
  ↓
Global Average Pooling
  ↓
Dropout (0.5)
  ↓
Dense(512, ReLU) + Dropout(0.3)
  ↓
Dense(256, ReLU) + Dropout(0.2)
  ↓
Dense(10, Softmax) → Output
```

## 📁 Proje Yapısı

```
SinirAglari_But/
├── data/
│   ├── original/          # Orijinal veri seti
│   └── augmented/         # Artırılmış veri seti
├── models/
│   ├── best_model.h5      # En iyi model
│   ├── final_model.h5     # Son model
│   ├── class_mapping.json # Sınıf mapping'leri
│   ├── training_log.csv   # Eğitim logları
│   └── *.png              # Değerlendirme görselleri
├── src/
│   ├── data_preprocessing.py    # Veri ön işleme
│   ├── model_training.py        # Model eğitimi
│   ├── model_evaluation.py      # Model değerlendirme
│   └── utils.py                 # Yardımcı fonksiyonlar
├── app/
│   ├── __init__.py
│   ├── routes.py                # Flask routes
│   ├── model_loader.py          # Model yükleme
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
├── templates/
│   └── index.html               # Ana sayfa
├── train.py                     # Eğitim scripti
├── app.py                       # Flask uygulaması
├── requirements.txt
└── README.md
```

## Teknolojiler

- **Deep Learning:** TensorFlow/Keras
- **Model:** EfficientNet-B3
- **Web Framework:** Flask
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Visualization:** Chart.js, Matplotlib, Seaborn
- **Data Processing:** NumPy, Pandas, PIL
- **Machine Learning:** Scikit-learn

## Değerlendirme Metrikleri

Model eğitimi sonrası aşağıdaki metrikler hesaplanır:

- **Accuracy:** Genel doğruluk
- **Precision:** Her sınıf için hassasiyet
- **Recall:** Her sınıf için duyarlılık
- **F1-Score:** Precision ve Recall'un harmonik ortalaması
- **Confusion Matrix:** Sınıflandırma karışıklık matrisi
- **ROC Curves:** Receiver Operating Characteristic eğrileri
- **Precision-Recall Curves:** PR eğrileri

Tüm metrikler ve görseller `models/` klasörüne kaydedilir.
