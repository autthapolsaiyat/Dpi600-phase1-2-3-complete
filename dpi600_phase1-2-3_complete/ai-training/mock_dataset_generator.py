"""
DPI-600 Mock Dataset Generator
==============================
สร้าง Mock Dataset สำหรับทดสอบ AI Training Pipeline
- สร้างภาพจำลองเม็ดยาพร้อม Logo
- รองรับหลาย Logo types
- สร้าง Labels อัตโนมัติ
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install Pillow numpy --quiet")
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np


# =====================================================
# DRUG PROFILE DEFINITIONS
# =====================================================
DRUG_PROFILES = {
    'lion': {
        'name_th': 'สิงโต',
        'name_en': 'Lion',
        'category': 'meth',
        'colors': ['#ff6b35', '#ff8c00', '#ffa500'],  # Orange shades
        'logo_text': '🦁',
        'logo_pattern': 'lion',
        'pill_shape': 'round',
        'size_range': (5.5, 6.5),  # mm
    },
    'wy': {
        'name_th': 'WY',
        'name_en': 'WY',
        'category': 'meth',
        'colors': ['#00ff88', '#00cc6a', '#00aa55'],  # Green shades
        'logo_text': 'WY',
        'logo_pattern': 'text',
        'pill_shape': 'round',
        'size_range': (5.8, 6.2),
    },
    '999': {
        'name_th': '999',
        'name_en': '999',
        'category': 'meth',
        'colors': ['#ff69b4', '#ff1493', '#db7093'],  # Pink shades
        'logo_text': '999',
        'logo_pattern': 'text',
        'pill_shape': 'round',
        'size_range': (5.5, 6.0),
    },
    'horse': {
        'name_th': 'ม้า',
        'name_en': 'Horse',
        'category': 'meth',
        'colors': ['#8b4513', '#a0522d', '#cd853f'],  # Brown shades
        'logo_text': '🐴',
        'logo_pattern': 'horse',
        'pill_shape': 'round',
        'size_range': (5.8, 6.3),
    },
    'r_mark': {
        'name_th': 'R',
        'name_en': 'R Mark',
        'category': 'meth',
        'colors': ['#ff0000', '#cc0000', '#990000'],  # Red shades
        'logo_text': 'R',
        'logo_pattern': 'text',
        'pill_shape': 'round',
        'size_range': (5.5, 6.2),
    },
    'star': {
        'name_th': 'ดาว',
        'name_en': 'Star',
        'category': 'meth',
        'colors': ['#ffd700', '#ffcc00', '#ffaa00'],  # Gold shades
        'logo_text': '★',
        'logo_pattern': 'star',
        'pill_shape': 'round',
        'size_range': (5.6, 6.1),
    },
    'eagle': {
        'name_th': 'นกอินทรี',
        'name_en': 'Eagle',
        'category': 'meth',
        'colors': ['#4169e1', '#0000ff', '#000080'],  # Blue shades
        'logo_text': '🦅',
        'logo_pattern': 'eagle',
        'pill_shape': 'round',
        'size_range': (5.7, 6.4),
    },
    'no_logo': {
        'name_th': 'ไม่มีโลโก้',
        'name_en': 'No Logo',
        'category': 'meth',
        'colors': ['#d3d3d3', '#c0c0c0', '#a9a9a9'],  # Gray shades
        'logo_text': '',
        'logo_pattern': 'none',
        'pill_shape': 'round',
        'size_range': (5.5, 6.5),
    },
    'ice': {
        'name_th': 'ไอซ์',
        'name_en': 'Ice/Crystal',
        'category': 'other',
        'colors': ['#e0ffff', '#b0e0e6', '#add8e6'],  # Crystal/ice colors
        'logo_text': '',
        'logo_pattern': 'crystal',
        'pill_shape': 'crystal',
        'size_range': (3.0, 15.0),  # Variable size
    },
    'heroin': {
        'name_th': 'เฮโรอีน',
        'name_en': 'Heroin',
        'category': 'other',
        'colors': ['#8b0000', '#654321', '#3d2314'],  # Dark brown/red
        'logo_text': '',
        'logo_pattern': 'none',
        'pill_shape': 'powder',
        'size_range': (0, 0),  # Powder
    },
}

# Distribution weights for realistic dataset
DISTRIBUTION_WEIGHTS = {
    'lion': 0.25,
    'wy': 0.20,
    '999': 0.15,
    'horse': 0.08,
    'r_mark': 0.05,
    'star': 0.05,
    'eagle': 0.05,
    'no_logo': 0.07,
    'ice': 0.08,
    'heroin': 0.02,
}


class MockDatasetGenerator:
    """Generate mock drug pill images for testing"""
    
    def __init__(self, output_dir="mock_dataset", image_size=(640, 480)):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_pill_image(self, profile_id, variation=0):
        """Create a mock pill image"""
        profile = DRUG_PROFILES[profile_id]
        
        # Create base image with scale plate background
        img = Image.new('RGB', self.image_size, color='#f5f5f5')
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines (scale plate simulation)
        grid_size = 50  # pixels per 10mm
        for x in range(0, self.image_size[0], grid_size):
            draw.line([(x, 0), (x, self.image_size[1])], fill='#cccccc', width=1)
        for y in range(0, self.image_size[1], grid_size):
            draw.line([(0, y), (self.image_size[0], y)], fill='#cccccc', width=1)
        
        # Draw QR code area (top-left corner)
        qr_size = 80
        draw.rectangle([10, 10, 10 + qr_size, 10 + qr_size], fill='white', outline='black')
        # Simplified QR pattern
        for i in range(8):
            for j in range(8):
                if random.random() > 0.5:
                    x = 15 + i * 9
                    y = 15 + j * 9
                    draw.rectangle([x, y, x + 7, y + 7], fill='black')
        
        # Draw pill based on shape
        center_x = self.image_size[0] // 2 + random.randint(-50, 50)
        center_y = self.image_size[1] // 2 + random.randint(-30, 30)
        
        if profile['pill_shape'] == 'round':
            self._draw_round_pill(draw, center_x, center_y, profile, variation)
        elif profile['pill_shape'] == 'crystal':
            self._draw_crystal(draw, center_x, center_y, profile, variation)
        elif profile['pill_shape'] == 'powder':
            self._draw_powder(draw, center_x, center_y, profile, variation)
        
        # Add some noise/texture
        img = self._add_texture(img)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            img = img.rotate(angle, fillcolor='#f5f5f5')
        
        return img
    
    def _draw_round_pill(self, draw, cx, cy, profile, variation):
        """Draw a round pill with logo"""
        # Size based on profile
        min_size, max_size = profile['size_range']
        size_mm = random.uniform(min_size, max_size)
        radius = int(size_mm * 8)  # Scale factor
        
        # Pill color with variation
        base_color = random.choice(profile['colors'])
        
        # Draw pill shadow
        shadow_offset = 3
        draw.ellipse([
            cx - radius + shadow_offset, 
            cy - radius + shadow_offset,
            cx + radius + shadow_offset, 
            cy + radius + shadow_offset
        ], fill='#888888')
        
        # Draw main pill
        draw.ellipse([
            cx - radius, cy - radius,
            cx + radius, cy + radius
        ], fill=base_color, outline='#333333', width=2)
        
        # Draw logo/text
        if profile['logo_text']:
            logo = profile['logo_text']
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                         int(radius * 0.8))
            except:
                font = ImageFont.load_default()
            
            # Get text size for centering
            bbox = draw.textbbox((0, 0), logo, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw text shadow
            draw.text((cx - text_width//2 + 1, cy - text_height//2 + 1), 
                     logo, fill='#00000066', font=font)
            # Draw text
            draw.text((cx - text_width//2, cy - text_height//2), 
                     logo, fill='#000000', font=font)
        
        # Draw score line (common on pills)
        if random.random() > 0.3:
            draw.line([(cx - radius + 5, cy), (cx + radius - 5, cy)], 
                     fill='#00000033', width=2)
    
    def _draw_crystal(self, draw, cx, cy, profile, variation):
        """Draw ice/crystal meth"""
        color = random.choice(profile['colors'])
        
        # Draw multiple crystal shards
        num_crystals = random.randint(3, 8)
        for _ in range(num_crystals):
            # Random position around center
            x = cx + random.randint(-60, 60)
            y = cy + random.randint(-40, 40)
            
            # Crystal size
            size = random.randint(15, 45)
            
            # Draw crystal shape (polygon)
            points = []
            num_points = random.randint(5, 8)
            for i in range(num_points):
                angle = (i / num_points) * 360
                r = size * (0.5 + random.random() * 0.5)
                px = x + r * np.cos(np.radians(angle))
                py = y + r * np.sin(np.radians(angle))
                points.append((px, py))
            
            draw.polygon(points, fill=color, outline='#ffffff')
    
    def _draw_powder(self, draw, cx, cy, profile, variation):
        """Draw powder (heroin)"""
        color = random.choice(profile['colors'])
        
        # Draw powder mound
        for _ in range(500):
            x = cx + random.gauss(0, 50)
            y = cy + random.gauss(0, 30)
            size = random.randint(1, 3)
            c = self._vary_color(color, 20)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=c)
    
    def _vary_color(self, hex_color, amount):
        """Vary a hex color slightly"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        r = max(0, min(255, r + random.randint(-amount, amount)))
        g = max(0, min(255, g + random.randint(-amount, amount)))
        b = max(0, min(255, b + random.randint(-amount, amount)))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _add_texture(self, img):
        """Add slight texture/noise to image"""
        img_array = np.array(img)
        noise = np.random.normal(0, 3, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def generate_dataset(self, num_images=1000, split_ratio=(0.8, 0.1, 0.1)):
        """
        Generate complete mock dataset with train/val/test split.
        
        Args:
            num_images: Total number of images to generate
            split_ratio: (train, val, test) ratios
        """
        print(f"🎲 Generating {num_images} mock images...")
        
        # Calculate images per class based on distribution
        images_per_class = {}
        for profile_id, weight in DISTRIBUTION_WEIGHTS.items():
            images_per_class[profile_id] = int(num_images * weight)
        
        # Adjust to match exact total
        diff = num_images - sum(images_per_class.values())
        if diff > 0:
            images_per_class['lion'] += diff
        
        # Create directories
        for split in ['train', 'val', 'test']:
            for profile_id in DRUG_PROFILES.keys():
                (self.output_dir / split / profile_id).mkdir(parents=True, exist_ok=True)
        
        # Generate images
        all_labels = []
        image_count = 0
        
        for profile_id, count in images_per_class.items():
            print(f"  Generating {count} images for {profile_id}...")
            
            for i in range(count):
                # Create image
                img = self.create_pill_image(profile_id, variation=i)
                
                # Determine split
                rand = random.random()
                if rand < split_ratio[0]:
                    split = 'train'
                elif rand < split_ratio[0] + split_ratio[1]:
                    split = 'val'
                else:
                    split = 'test'
                
                # Save image
                filename = f"{profile_id}_{i:04d}.jpg"
                filepath = self.output_dir / split / profile_id / filename
                img.save(filepath, 'JPEG', quality=90)
                
                # Create label entry
                label_entry = {
                    'filename': filename,
                    'filepath': str(filepath),
                    'split': split,
                    'label_id': profile_id,
                    'label_th': DRUG_PROFILES[profile_id]['name_th'],
                    'label_en': DRUG_PROFILES[profile_id]['name_en'],
                    'category': DRUG_PROFILES[profile_id]['category'],
                    'generated_at': datetime.now().isoformat()
                }
                all_labels.append(label_entry)
                
                image_count += 1
                if image_count % 100 == 0:
                    print(f"    Generated {image_count}/{num_images} images...")
        
        # Save labels
        labels_file = self.output_dir / 'labels.json'
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(all_labels, f, ensure_ascii=False, indent=2)
        
        # Create summary
        summary = self._create_summary(all_labels)
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Total images: {len(all_labels)}")
        print(f"   Train: {len([l for l in all_labels if l['split'] == 'train'])}")
        print(f"   Val: {len([l for l in all_labels if l['split'] == 'val'])}")
        print(f"   Test: {len([l for l in all_labels if l['split'] == 'test'])}")
        print(f"   Output: {self.output_dir}")
        
        return all_labels
    
    def _create_summary(self, labels):
        """Create dataset summary"""
        summary = {
            'name': 'DPI-600 Mock Drug Profile Dataset',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'total_images': len(labels),
            'splits': {
                'train': len([l for l in labels if l['split'] == 'train']),
                'val': len([l for l in labels if l['split'] == 'val']),
                'test': len([l for l in labels if l['split'] == 'test']),
            },
            'classes': {}
        }
        
        for profile_id in DRUG_PROFILES.keys():
            count = len([l for l in labels if l['label_id'] == profile_id])
            summary['classes'][profile_id] = {
                'name_th': DRUG_PROFILES[profile_id]['name_th'],
                'name_en': DRUG_PROFILES[profile_id]['name_en'],
                'count': count,
                'percentage': f"{count / len(labels) * 100:.1f}%"
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="DPI-600 Mock Dataset Generator")
    parser.add_argument("--output", "-o", default="mock_dataset",
                       help="Output directory")
    parser.add_argument("--num-images", "-n", type=int, default=1000,
                       help="Number of images to generate")
    parser.add_argument("--size", "-s", type=int, nargs=2, default=[640, 480],
                       help="Image size (width height)")
    
    args = parser.parse_args()
    
    generator = MockDatasetGenerator(
        output_dir=args.output,
        image_size=tuple(args.size)
    )
    
    generator.generate_dataset(num_images=args.num_images)


if __name__ == "__main__":
    main()
