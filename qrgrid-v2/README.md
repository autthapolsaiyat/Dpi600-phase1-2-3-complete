# 📱 DPI-600 QRGrid Capture v2

ระบบถ่ายภาพเม็ดยาพร้อม Scale Plate Detection และ QR Code Recognition

## 🚀 Features

### ✅ QR Code Detection
- ตรวจจับ QR Code อัตโนมัติ (jsQR library)
- รองรับ 4 ขนาด Scale Plate

### ✅ Smart Frame
- 🔴 **แดง** = ไม่พบ QR Code
- 🟡 **เหลือง** = กำลังค้นหา
- 🟢 **เขียว** = พร้อมถ่าย

### ✅ Scale Plate Database

| QR Code | Plate ID | Grid Size | สำหรับ |
|---------|----------|-----------|--------|
| `DPI600-S01-5MM` | SCALE-01 | 5mm | ยาบ้าเม็ดเล็ก |
| `DPI600-S02-10MM` | SCALE-02 | 10mm | ยาบ้าทั่วไป |
| `DPI600-S03-20MM` | SCALE-03 | 20mm | ไอซ์/ยาเม็ดใหญ่ |
| `DPI600-S04-50MM` | SCALE-04 | 50mm | ของกลางขนาดใหญ่ |

### ✅ Auto Data Capture
- บันทึก GPS อัตโนมัติ
- บันทึก Timestamp
- บันทึก Plate Info
- บันทึก Resolution

## 📋 วิธีใช้งาน

### 1. รันบน Local Server

```bash
cd qrgrid-v2
python3 -m http.server 8888
```

เปิด browser: http://localhost:8888

### 2. รันบนมือถือ (WiFi เดียวกัน)

```bash
# หา IP ของเครื่อง
ipconfig getifaddr en0  # Mac
# หรือ
hostname -I  # Linux

# รัน server
python3 -m http.server 8888 --bind 0.0.0.0
```

เปิดบนมือถือ: http://[IP]:8888

## 🔳 สร้าง QR Code สำหรับ Scale Plate

ใช้เว็บ https://www.qr-code-generator.com/ หรือ library สร้าง QR Code ที่มีข้อความ:

- `DPI600-S01-5MM` สำหรับ Plate 5mm
- `DPI600-S02-10MM` สำหรับ Plate 10mm
- `DPI600-S03-20MM` สำหรับ Plate 20mm
- `DPI600-S04-50MM` สำหรับ Plate 50mm

## 📐 ออกแบบ Scale Plate

```
┌────────────────────────────────────┐
│  ┌──────────┐                      │
│  │ QR CODE  │   DPI-600            │
│  │          │   SCALE-01           │
│  └──────────┘   5mm Grid           │
│                                    │
│  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤            │
│  │ │ │ │ │ │ │ │ │ │ │  ← 5mm each │
│  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤            │
│  │ │ │ │ │ │ │ │ │ │ │            │
│  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤            │
│  │ │ │ │ │ │ │ │ │ │ │            │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘            │
│                                    │
│  ┌────────────────────┐           │
│  │   วางเม็ดยาที่นี่    │           │
│  └────────────────────┘           │
│                                    │
└────────────────────────────────────┘
```

## 📁 โครงสร้างไฟล์

```
qrgrid-v2/
├── index.html      # Main PWA
├── manifest.json   # PWA Manifest
├── sw.js          # Service Worker
└── README.md      # Documentation
```

## 📊 ข้อมูลที่บันทึกในแต่ละภาพ

```json
{
  "id": "IMG-1702195200000",
  "image": "data:image/jpeg;base64,...",
  "timestamp": "2024-12-10T07:00:00.000Z",
  "plate": {
    "id": "SCALE-01",
    "name": "5mm Grid",
    "gridSize": 5,
    "precision": "±0.1mm"
  },
  "qrCode": "DPI600-S01-5MM",
  "gps": {
    "lat": 13.918210,
    "lng": 100.346145,
    "accuracy": 10
  },
  "metadata": {
    "resolution": "1920x1080",
    "officerId": "OFF-001"
  }
}
```

## 🔄 Next Steps

1. **Phase 2**: เก็บภาพเม็ดยา 1,000+ ภาพ
2. **Phase 3**: Labeling ภาพตาม Logo/หัวปั๊ม
3. **Phase 4**: Train AI Model บน Azure ML
4. **Phase 5**: เชื่อมต่อ AI กับ PWA

## 📞 Support

- Project: DPI-600 Drug Profile AI
- Developer: Saengvith Science Co., Ltd.
- Version: 2.0.0
