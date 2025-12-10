# 🏷️ DPI-600 Drug Label Studio

ระบบ Labeling สำหรับติดป้ายภาพเม็ดยาตาม Logo/หัวปั๊ม

## 📦 ไฟล์ในชุดนี้

```
labeling-studio/
├── index.html          # Web-based Labeling Interface
├── augmentation.py     # Data Augmentation Tool
├── import_tool.py      # Import จาก QRGrid PWA
└── README.md           # Documentation
```

## 🚀 Quick Start

### 1. เปิด Labeling Interface

```bash
cd labeling-studio
python3 -m http.server 8889
```

เปิด Browser: **http://localhost:8889**

### 2. Import ภาพจาก QRGrid PWA

```bash
# Export จาก Browser Console (ใน QRGrid PWA)
# พิมพ์: localStorage.getItem('qrgrid_photos')
# Copy ข้อความที่ได้ไปใส่ไฟล์ qrgrid_export.json

# แล้วรัน Import
python3 import_tool.py json qrgrid_export.json --output raw
```

### 3. Data Augmentation

```bash
# เพิ่มจำนวนภาพ 10x
python3 augmentation.py augment --source raw --labels labels.json --output augmented
```

---

## 🏷️ Labels ที่รองรับ (Default)

### ยาบ้า (Methamphetamine)

| ID | ไทย | English | สี | Shortcut |
|----|-----|---------|-----|----------|
| lion | สิงโต | Lion | 🟠 | 1 |
| wy | WY | WY | 🟢 | 2 |
| 999 | 999 | 999 | 🩷 | 3 |
| horse | ม้า | Horse | 🟤 | 4 |
| r_mark | R | R Mark | 🔴 | 5 |
| star | ดาว | Star | 🟡 | 6 |
| eagle | นกอินทรี | Eagle | 🔵 | 7 |
| no_logo | ไม่มีโลโก้ | No Logo | ⬜ | 8 |

### ยาเสพติดอื่น

| ID | ไทย | English | สี | Shortcut |
|----|-----|---------|-----|----------|
| ice | ไอซ์ | Ice/Crystal | 💎 | 9 |
| heroin | เฮโรอีน | Heroin | 🟤 | - |
| cocaine | โคเคน | Cocaine | ⚪ | - |
| ecstasy | ยาอี | Ecstasy | 🩷 | - |
| ketamine | เคตามีน | Ketamine | 🟣 | - |
| unknown | ไม่ทราบ | Unknown | ⬛ | - |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` หรือ `D` | ภาพถัดไป |
| `←` หรือ `A` | ภาพก่อนหน้า |
| `1-9` | เลือก Label |
| `Backspace` | ลบ Label |

---

## 📊 Export Formats

### 1. Dataset Metadata (JSON)

```json
{
  "metadata": {
    "name": "DPI-600 Drug Profile Dataset",
    "totalImages": 1000,
    "labeledImages": 950,
    "labels": [...]
  },
  "images": [
    {
      "filename": "img_001.jpg",
      "label": "lion",
      "dimensions": {"width": 1920, "height": 1080}
    }
  ]
}
```

### 2. Labels Only (JSON)

```json
[
  {"filename": "img_001.jpg", "label": "lion", "labelTh": "สิงโต"},
  {"filename": "img_002.jpg", "label": "wy", "labelTh": "WY"}
]
```

### 3. Labels CSV

```csv
filename,label_id,label_th,label_en
img_001.jpg,lion,สิงโต,Lion
img_002.jpg,wy,WY,WY
```

---

## 🔄 Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  📷 QRGrid   │ ──► │ 🏷️ Labeling │ ──► │ 📦 Dataset   │
│  PWA v2      │     │   Studio     │     │   Export     │
│              │     │              │     │              │
│ ถ่ายภาพ      │     │ ติด Label    │     │ train/val/   │
│ + Scale Plate│     │ ตาม Logo     │     │ test split   │
└──────────────┘     └──────────────┘     └──────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
   qrgrid_photos      labels.json         Ready for
   (localStorage)                         AI Training
```

---

## 📁 Dataset Structure (สำหรับ Training)

```
dataset/
├── train/              # 80% of data
│   ├── lion/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── wy/
│   ├── 999/
│   └── ...
├── val/                # 10% of data
│   ├── lion/
│   ├── wy/
│   └── ...
├── test/               # 10% of data
│   ├── lion/
│   ├── wy/
│   └── ...
└── labels.json         # All labels
```

---

## 🔧 Data Augmentation

### Augmentation Types

| Type | Description | Factor |
|------|-------------|--------|
| rotate_90 | หมุน 90° | 1x |
| rotate_180 | หมุน 180° | 1x |
| rotate_270 | หมุน 270° | 1x |
| flip_h | พลิกแนวนอน | 1x |
| flip_v | พลิกแนวตั้ง | 1x |
| bright_up | เพิ่มความสว่าง | 1x |
| bright_down | ลดความสว่าง | 1x |
| contrast_up | เพิ่ม contrast | 1x |
| contrast_down | ลด contrast | 1x |
| noise | เพิ่ม noise | 1x |
| crop_center | Crop กลาง 90% | 1x |

**Total: 12x augmentation** (1 original + 11 augmented)

### Usage

```bash
# Initialize dataset structure
python3 augmentation.py init

# Augment images
python3 augmentation.py augment --source raw --labels labels.json --output augmented

# Split into train/val/test
python3 augmentation.py split --source augmented
```

---

## 📈 Target Dataset Size

| Phase | Original | Augmented | Total |
|-------|----------|-----------|-------|
| Initial | 100 | 1,200 | 1,200 |
| Phase 1 | 500 | 6,000 | 6,000 |
| Phase 2 | 1,000 | 12,000 | 12,000 |
| Production | 2,000+ | 24,000+ | 24,000+ |

---

## 🎯 Next Steps

- [x] Phase 1: QRGrid PWA v2
- [x] Phase 2: Labeling Studio
- [ ] Phase 3: Collect 1,000+ images
- [ ] Phase 4: Train AI Model (Azure ML)
- [ ] Phase 5: Integration

---

## 📞 Support

- **Project:** DPI-600 Drug Profile AI
- **Developer:** Saengvith Science Co., Ltd.
- **Version:** 2.0.0
