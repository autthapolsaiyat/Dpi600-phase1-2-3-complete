"""
DPI-600 QRGrid Import Tool
==========================
นำเข้าภาพจาก QRGrid PWA เข้าสู่ระบบ Labeling
"""

import os
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime


def decode_base64_image(data_url, output_path):
    """Decode base64 data URL to image file"""
    # Remove data URL prefix
    if ',' in data_url:
        header, data = data_url.split(',', 1)
    else:
        data = data_url
    
    # Decode and save
    image_data = base64.b64decode(data)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    return output_path


def import_from_qrgrid_json(json_file, output_dir="raw"):
    """
    Import images from QRGrid PWA export JSON.
    
    Expected JSON format:
    [
        {
            "id": "IMG-xxx",
            "image": "data:image/jpeg;base64,...",
            "timestamp": "2024-12-10T...",
            "plate": { "id": "SCALE-01", ... },
            "gps": { "lat": ..., "lng": ... }
        }
    ]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} images in {json_file}")
    
    imported = []
    for item in data:
        # Generate filename
        img_id = item.get('id', f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        filename = f"{img_id}.jpg"
        filepath = output_path / filename
        
        # Decode and save image
        if 'image' in item:
            decode_base64_image(item['image'], filepath)
            
            # Create metadata
            metadata = {
                'filename': filename,
                'original_id': item.get('id'),
                'timestamp': item.get('timestamp'),
                'plate_id': item.get('plate', {}).get('id'),
                'plate_scale': item.get('plate', {}).get('gridSize'),
                'gps_lat': item.get('gps', {}).get('lat'),
                'gps_lng': item.get('gps', {}).get('lng'),
                'label': None  # To be labeled
            }
            imported.append(metadata)
            print(f"  ✓ Imported: {filename}")
    
    # Save metadata
    metadata_file = output_path / 'import_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(imported, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Imported {len(imported)} images to {output_path}")
    print(f"   Metadata saved to: {metadata_file}")
    
    return imported


def import_from_localstorage(browser_export_file, output_dir="raw"):
    """
    Import from browser localStorage export.
    
    To export from browser console:
    localStorage.getItem('qrgrid_photos')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read file
    with open(browser_export_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Maybe it's a raw string from localStorage
        data = json.loads(content.strip('"').replace('\\"', '"'))
    
    return import_from_qrgrid_json_data(data, output_dir)


def import_from_qrgrid_json_data(data, output_dir="raw"):
    """Import from already parsed JSON data"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(data, list):
        data = [data]
    
    print(f"Processing {len(data)} images...")
    
    imported = []
    for idx, item in enumerate(data):
        # Generate filename
        img_id = item.get('id', f"img_{idx:04d}")
        filename = f"{img_id}.jpg"
        filepath = output_path / filename
        
        # Decode and save image
        if 'image' in item:
            try:
                decode_base64_image(item['image'], filepath)
                
                # Create metadata
                metadata = {
                    'filename': filename,
                    'original_id': item.get('id'),
                    'timestamp': item.get('timestamp'),
                    'plate_id': item.get('plate', {}).get('id') if item.get('plate') else None,
                    'gps': item.get('gps'),
                    'label': None
                }
                imported.append(metadata)
                print(f"  ✓ {filename}")
            except Exception as e:
                print(f"  ✗ Error processing {img_id}: {e}")
    
    # Save metadata
    if imported:
        metadata_file = output_path / 'import_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(imported, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Imported {len(imported)} images")
    
    return imported


def import_from_folder(source_dir, output_dir="raw"):
    """Import images from a folder"""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in source.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} images in {source_dir}")
    
    imported = []
    for img in images:
        # Copy to output
        import shutil
        dest = output / img.name
        shutil.copy(img, dest)
        
        imported.append({
            'filename': img.name,
            'original_path': str(img),
            'label': None
        })
        print(f"  ✓ {img.name}")
    
    # Save metadata
    metadata_file = output / 'import_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(imported, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Imported {len(imported)} images to {output}")
    
    return imported


def create_labeling_template(images_dir, output_file="labels_template.json"):
    """Create a template JSON for labeling"""
    images_path = Path(images_dir)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f.name for f in images_path.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    template = [
        {
            'filename': img,
            'label': None,
            'label_th': None,
            'notes': ''
        }
        for img in sorted(images)
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created labeling template: {output_file}")
    print(f"   Total images: {len(template)}")
    
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPI-600 QRGrid Import Tool")
    parser.add_argument("command", choices=["json", "folder", "template"],
                       help="Import command")
    parser.add_argument("source", help="Source file or directory")
    parser.add_argument("--output", "-o", default="raw",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "json":
        import_from_qrgrid_json(args.source, args.output)
    elif args.command == "folder":
        import_from_folder(args.source, args.output)
    elif args.command == "template":
        create_labeling_template(args.source, args.output)
