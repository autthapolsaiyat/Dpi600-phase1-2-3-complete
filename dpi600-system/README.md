# 🔬 DPI-600 Drug Profile Intelligence System

## ระบบวิเคราะห์โปรไฟล์ยาเสพติดด้วย AI

**Version:** 2.0  
**Release Date:** December 2568 (2025)  
**Developer:** Saengvith Science Co., Ltd.  
**Contact:** Autthapol Saiyat (Boy) | 085-070-9938 | LINE: boy_saiyat

---

## 📁 File Structure

```
dpi600-system/
├── index.html                              # Main Portal (Entry Point)
├── DPI600_Investigation_Center.html        # Case Analysis + AI Matching
├── DPI600_Pill_Fingerprint_Matching.html   # Pill Shape Comparison + Overlay
├── DPI600_Profile_Manager.html             # Drug Profile Management
├── DPI600_Case_History_Timeline.html       # Case History & Timeline
├── DPI600_Case_Detail.html                 # Case Detail View
├── DPI600_PDF_Report_Template.html         # PDF Report Generator
├── DPI600_Executive_Presentation_v2.html   # Executive Presentation
├── DPI600_Quotation_Interactive.html       # Project Quotation
├── dpi600_user_portal_v2.html              # User Portal (Legacy)
└── sql/
    ├── DPI600_Database_Schema_v2.sql       # Main Database Schema
    └── DPI600_Pill_Profile_Schema.sql      # Pill Profile Schema Extension
```

---

## 🚀 Quick Start

### Option 1: Local Development
```bash
# Simply open in browser
open index.html
# or
python -m http.server 8000
# then visit http://localhost:8000
```

### Option 2: Azure Static Web Apps
```bash
# Using Azure CLI
az staticwebapp create \
  --name dpi600-system \
  --resource-group your-resource-group \
  --source ./dpi600-system \
  --location "East Asia"
```

### Option 3: Nginx/Apache
```bash
# Copy to web root
sudo cp -r dpi600-system/* /var/www/html/dpi600/
```

---

## 🔐 Demo Credentials

```
Username: admin
Password: admin123
```

---

## 📊 Features

### 🔍 Investigation Center
- Upload drug images
- Input chemical profile data
- AI-powered organization matching
- Geographic heatmap visualization
- Confidence scoring system

### 🔬 Pill Fingerprint Matching (NEW)
- Visual shape comparison
- Image overlay with adjustable blend modes
- Dimension analysis (±5% tolerance)
- Chemical signature comparison
- Same batch detection
- Drug profile creation

### 🆔 Profile Manager (NEW)
- View all drug profiles
- Grid and list views
- Filter by logo, organization, status
- Profile detail modal with map
- Export capabilities

### 📅 Case History & Timeline
- Activity timeline with filters
- Monthly trend charts
- Organization statistics
- Geographic distribution map

### 📄 PDF Report Generator
- Print-ready format
- Case overview and details
- Chemical analysis visualization
- Organization match results
- Signature section

---

## 🗄️ Database Schema

### Main Tables (15+)
- `organizations` - Drug trafficking networks
- `drug_cases` - Seizure cases
- `pill_profiles` - Drug pill fingerprints
- `case_organization_matches` - AI match results
- `case_profile_matches` - Pill profile matches
- `production_batches` - Batch tracking
- `drug_logos` - Known logos
- `provinces` - Thai provinces
- `forensic_centers` - 11 forensic centers
- `users` - System users
- `audit_log` - Activity logging

### Key Functions
- `calculate_chemical_similarity()` - Compare chemical profiles
- `calculate_match_score()` - Compute overall match score
- `calculate_visual_similarity()` - Compare pill dimensions
- `generate_profile_id()` - Auto-generate profile IDs

### Views
- `v_case_top_matches` - Top organization matches per case
- `v_organization_stats` - Organization statistics
- `v_profile_summary` - Profile summary with stats
- `v_batch_timeline` - Batch tracking timeline

---

## 🔧 Database Installation

```bash
# PostgreSQL with PostGIS
psql -U postgres -d dpi600 -f sql/DPI600_Database_Schema_v2.sql
psql -U postgres -d dpi600 -f sql/DPI600_Pill_Profile_Schema.sql
```

Required Extensions:
- `postgis` - Geographic data
- `pg_trgm` - Text similarity search

---

## 🎨 UI Features

All pages include:
- ✅ Dark/Light theme toggle
- ✅ Login gate with demo credentials
- ✅ Responsive design (mobile-friendly)
- ✅ Interactive maps (Leaflet.js)
- ✅ Charts (Chart.js)
- ✅ Cyberpunk aesthetic

---

## 📱 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## 📞 Support

**Saengvith Science Co., Ltd.**  
Technology & AI Program Director: Autthapol Saiyat (Boy)

- 📱 Phone: 085-070-9938
- 💬 LINE: boy_saiyat
- 📧 Email: [Contact for email]

---

## 📄 License

Proprietary - Office of Police Forensic Science, Royal Thai Police

© 2025 Saengvith Science Co., Ltd. All rights reserved.
