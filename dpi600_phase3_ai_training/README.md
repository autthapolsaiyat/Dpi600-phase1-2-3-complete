# 🧠 DPI-600 Drug Logo Recognition - AI Training Pipeline

ระบบฝึกสอน AI สำหรับจำแนกโลโก้/ตราประทับบนเม็ดยาเสพติด

## 📋 สารบัญ

- [ภาพรวม](#ภาพรวม)
- [โครงสร้างไฟล์](#โครงสร้างไฟล์)
- [การติดตั้ง](#การติดตั้ง)
- [การใช้งาน](#การใช้งาน)
- [Azure ML](#azure-ml)
- [API Server](#api-server)

---

## 📖 ภาพรวม

### เป้าหมาย
- ความแม่นยำ ≥ 95%
- รองรับ 10 ประเภทโลโก้
- ประมวลผล < 100ms ต่อภาพ

### Model Architecture
- **Base Model:** EfficientNet-B0 (Transfer Learning)
- **Input Size:** 224 x 224 pixels
- **Output:** 10 classes

### Classes

| ID | รหัส | ไทย | English |
|----|------|-----|---------|
| 0 | lion | สิงโต | Lion |
| 1 | wy | WY | WY |
| 2 | 999 | 999 | 999 |
| 3 | horse | ม้า | Horse |
| 4 | r_mark | R | R Mark |
| 5 | star | ดาว | Star |
| 6 | eagle | นกอินทรี | Eagle |
| 7 | no_logo | ไม่มีโลโก้ | No Logo |
| 8 | ice | ไอซ์ | Ice/Crystal |
| 9 | heroin | เฮโรอีน | Heroin |

---

## 📁 โครงสร้างไฟล์

```
ai-training/
├── train_model.py          # Main training script
├── inference.py            # Inference & API server
├── mock_dataset_generator.py   # Generate mock dataset
├── azure_ml_pipeline.py    # Azure ML integration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
├── docs/
│   └── SOP_Data_Collection_TH.md   # คู่มือเก็บข้อมูล
└── README.md               # This file
```

---

## 🚀 การติดตั้ง

### Option 1: Local Installation

```bash
# Clone หรือ Download files
cd ai-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build training image
docker build --target training -t dpi600-training .

# Build inference image
docker build --target inference -t dpi600-inference .
```

---

## 💻 การใช้งาน

### 1. Generate Mock Dataset (ทดสอบ)

```bash
# สร้าง 1000 ภาพจำลอง
python mock_dataset_generator.py --output mock_dataset --num-images 1000

# Output structure:
# mock_dataset/
# ├── train/
# │   ├── lion/
# │   ├── wy/
# │   └── ...
# ├── val/
# ├── test/
# ├── labels.json
# └── dataset_summary.json
```

### 2. Train Model

```bash
# Basic training
python train_model.py \
    --data-dir ./mock_dataset \
    --output ./output \
    --epochs 50 \
    --batch-size 32

# Fine-tune only classifier
python train_model.py \
    --data-dir ./dataset \
    --output ./output \
    --freeze-backbone

# Different model architecture
python train_model.py \
    --data-dir ./dataset \
    --model mobilenet_v3 \
    --output ./output
```

### 3. Inference

```bash
# Single image
python inference.py \
    --model ./output/checkpoints/best_model.pth \
    --image ./test_image.jpg

# Directory of images
python inference.py \
    --model ./output/checkpoints/best_model.pth \
    --directory ./test_images \
    --output results.json

# Start API server
python inference.py \
    --model ./output/checkpoints/best_model.pth \
    --api \
    --port 5000
```

---

## ☁️ Azure ML

### Setup Azure ML Workspace

1. สร้าง Azure ML Workspace ใน Azure Portal
2. ตั้งค่า Environment Variables:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="dpi600-rg"
export AZURE_WORKSPACE="dpi600-mlworkspace"
```

3. Generate Azure ML files:

```bash
python azure_ml_pipeline.py --output ./azure_ml --create-config
```

### Submit Training Job

```bash
cd azure_ml

# Install Azure ML SDK
pip install azure-ai-ml azure-identity

# Submit job
python submit_job.py --data-path ../dataset --epochs 50
```

### Monitor Training

- เข้า Azure ML Studio: https://ml.azure.com
- ดู Experiments > dpi600-drug-logo-recognition
- Track metrics: accuracy, loss

---

## 🌐 API Server

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single image prediction |
| POST | `/predict/batch` | Batch prediction |

### Example: cURL

```bash
# Health check
curl http://localhost:5000/health

# Predict single image
curl -X POST -F "image=@pill.jpg" http://localhost:5000/predict

# Batch prediction
curl -X POST \
    -F "images=@pill1.jpg" \
    -F "images=@pill2.jpg" \
    http://localhost:5000/predict/batch
```

### Example: Python

```python
import requests

# Single prediction
with open('pill.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    print(response.json())

# Expected output:
# {
#     "image": "pill.jpg",
#     "top_prediction": {
#         "rank": 1,
#         "label_id": "lion",
#         "label_th": "สิงโต",
#         "confidence": 97.5
#     },
#     "all_predictions": [...]
# }
```

---

## 🐳 Docker Commands

```bash
# Training
docker run --gpus all \
    -v $(pwd)/dataset:/app/data \
    -v $(pwd)/models:/app/models \
    dpi600-training \
    python train_model.py -d /app/data -o /app/models

# Inference API
docker run --gpus all \
    -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    dpi600-inference

# Generate mock dataset
docker run \
    -v $(pwd)/mock_data:/app/data \
    dpi600-training \
    python mock_dataset_generator.py -o /app/data -n 500
```

---

## 📊 Expected Results

### Training Metrics (Mock Dataset)

| Metric | Value |
|--------|-------|
| Train Accuracy | ~98% |
| Validation Accuracy | ~95% |
| Test Accuracy | ~94% |
| Training Time | ~30 min (GPU) |

### Real Dataset Target

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 95% |
| Precision | ≥ 94% |
| Recall | ≥ 93% |
| F1-Score | ≥ 93% |

---

## 📝 Data Collection Workflow

```
┌─────────────────┐
│  QRGrid PWA v2  │ ← ถ่ายภาพด้วย Scale Plate
└────────┬────────┘
         │ Export JSON
         ▼
┌─────────────────┐
│  Import Tool    │ ← import_tool.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Label Studio   │ ← ติด Label (1-9)
└────────┬────────┘
         │ Export labels.json
         ▼
┌─────────────────┐
│  Augmentation   │ ← augmentation.py (12x)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train Model    │ ← train_model.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deploy API     │ ← inference.py --api
└─────────────────┘
```

---

## 🔧 Troubleshooting

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA toolkit or use CPU:
python train_model.py --device cpu
```

### Out of Memory
```bash
# Reduce batch size
python train_model.py --batch-size 16

# Or use smaller model
python train_model.py --model mobilenet_v3
```

### Import Error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📄 License

DPI-600 Drug Profile AI System  
© 2024 สำนักงานพิสูจน์หลักฐานตำรวจ / Saengvith Science Co., Ltd.

---

## 📞 Contact

- Technical: dpi600-support@police.go.th
- Developer: Saengvith Science Co., Ltd.
