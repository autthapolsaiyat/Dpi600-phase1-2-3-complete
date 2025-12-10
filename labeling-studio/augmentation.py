"""
DPI-600 Data Augmentation Tool
==============================
เพิ่มจำนวนภาพด้วยเทคนิค Data Augmentation
- Rotation
- Flip
- Brightness adjustment
- Contrast adjustment
- Zoom/Crop
- Noise addition
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install Pillow numpy --quiet")
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np


class DataAugmentor:
    """Data Augmentation for drug pill images"""
    
    def __init__(self, output_dir="augmented"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def rotate(self, img, angle):
        """Rotate image by angle degrees"""
        return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    def flip_horizontal(self, img):
        """Flip image horizontally"""
        return ImageOps.mirror(img)
    
    def flip_vertical(self, img):
        """Flip image vertically"""
        return ImageOps.flip(img)
    
    def adjust_brightness(self, img, factor):
        """Adjust brightness. factor > 1 = brighter, < 1 = darker"""
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, img, factor):
        """Adjust contrast. factor > 1 = more contrast"""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, img, factor):
        """Adjust color saturation"""
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def add_gaussian_noise(self, img, mean=0, std=10):
        """Add Gaussian noise to image"""
        img_array = np.array(img)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def crop_center(self, img, crop_percent=0.9):
        """Crop center of image"""
        width, height = img.size
        new_width = int(width * crop_percent)
        new_height = int(height * crop_percent)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        return img.crop((left, top, left + new_width, top + new_height))
    
    def random_crop(self, img, crop_percent=0.85):
        """Random crop of image"""
        width, height = img.size
        new_width = int(width * crop_percent)
        new_height = int(height * crop_percent)
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        return img.crop((left, top, left + new_width, top + new_height))
    
    def blur(self, img, radius=1):
        """Apply Gaussian blur"""
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    def sharpen(self, img):
        """Sharpen image"""
        return img.filter(ImageFilter.SHARPEN)
    
    def augment_single(self, img_path, augmentations=None):
        """
        Augment a single image with specified augmentations.
        
        Args:
            img_path: Path to source image
            augmentations: List of augmentation configs, or None for default
            
        Returns:
            List of (augmented_image, augmentation_name) tuples
        """
        img = Image.open(img_path).convert('RGB')
        results = []
        
        if augmentations is None:
            # Default augmentations
            augmentations = [
                ('original', {}),
                ('rotate_90', {'rotate': 90}),
                ('rotate_180', {'rotate': 180}),
                ('rotate_270', {'rotate': 270}),
                ('flip_h', {'flip_h': True}),
                ('flip_v', {'flip_v': True}),
                ('bright_up', {'brightness': 1.2}),
                ('bright_down', {'brightness': 0.8}),
                ('contrast_up', {'contrast': 1.3}),
                ('contrast_down', {'contrast': 0.7}),
                ('noise', {'noise': True}),
                ('crop_center', {'crop_center': 0.9}),
            ]
        
        for name, config in augmentations:
            aug_img = img.copy()
            
            if 'rotate' in config:
                aug_img = self.rotate(aug_img, config['rotate'])
            if config.get('flip_h'):
                aug_img = self.flip_horizontal(aug_img)
            if config.get('flip_v'):
                aug_img = self.flip_vertical(aug_img)
            if 'brightness' in config:
                aug_img = self.adjust_brightness(aug_img, config['brightness'])
            if 'contrast' in config:
                aug_img = self.adjust_contrast(aug_img, config['contrast'])
            if 'saturation' in config:
                aug_img = self.adjust_saturation(aug_img, config['saturation'])
            if config.get('noise'):
                aug_img = self.add_gaussian_noise(aug_img)
            if 'crop_center' in config:
                aug_img = self.crop_center(aug_img, config['crop_center'])
            if 'blur' in config:
                aug_img = self.blur(aug_img, config['blur'])
            if config.get('sharpen'):
                aug_img = self.sharpen(aug_img)
            
            results.append((aug_img, name))
        
        return results
    
    def augment_dataset(self, source_dir, labels_file=None, multiplier=10):
        """
        Augment entire dataset.
        
        Args:
            source_dir: Directory containing source images
            labels_file: JSON file with labels (optional)
            multiplier: Target augmentation multiplier (e.g., 10 = 10x more images)
        """
        source_path = Path(source_dir)
        
        # Load labels if provided
        labels = {}
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                labels = {item['filename']: item['label'] for item in labels_data}
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = [f for f in source_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(images)} images in {source_dir}")
        
        # Create output directories by label
        augmented_labels = []
        total_generated = 0
        
        for img_path in images:
            label = labels.get(img_path.name, 'unknown')
            label_dir = self.output_dir / label
            label_dir.mkdir(exist_ok=True)
            
            # Generate augmentations
            augmented = self.augment_single(img_path)
            
            for aug_img, aug_name in augmented:
                # Save augmented image
                output_name = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                output_path = label_dir / output_name
                aug_img.save(output_path, quality=95)
                
                augmented_labels.append({
                    'filename': output_name,
                    'original': img_path.name,
                    'augmentation': aug_name,
                    'label': label
                })
                total_generated += 1
        
        # Save augmented labels
        labels_output = self.output_dir / 'augmented_labels.json'
        with open(labels_output, 'w', encoding='utf-8') as f:
            json.dump(augmented_labels, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Augmentation complete!")
        print(f"   Original images: {len(images)}")
        print(f"   Augmented images: {total_generated}")
        print(f"   Multiplier achieved: {total_generated / len(images):.1f}x")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Labels saved to: {labels_output}")
        
        return augmented_labels


def create_dataset_structure(base_dir="dataset"):
    """Create standard dataset directory structure"""
    base = Path(base_dir)
    
    # Standard ML dataset structure
    dirs = [
        base / "train",
        base / "val", 
        base / "test",
        base / "raw",
        base / "augmented"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme = base / "README.md"
    readme.write_text("""# DPI-600 Drug Profile Dataset

## Directory Structure

```
dataset/
├── raw/           # Original images (before augmentation)
├── augmented/     # Augmented images (organized by label)
├── train/         # Training set (80%)
├── val/           # Validation set (10%)
├── test/          # Test set (10%)
└── labels.json    # Label annotations
```

## Labels

| Label ID | Thai | English | Category |
|----------|------|---------|----------|
| lion | สิงโต | Lion | Meth |
| wy | WY | WY | Meth |
| 999 | 999 | 999 | Meth |
| horse | ม้า | Horse | Meth |
| ice | ไอซ์ | Ice/Crystal | Other |
| heroin | เฮโรอีน | Heroin | Other |

## Usage

```python
from augmentation import DataAugmentor

# Augment dataset
augmentor = DataAugmentor(output_dir="augmented")
augmentor.augment_dataset("raw", "labels.json", multiplier=10)
```
""")
    
    print(f"✅ Created dataset structure at: {base}")
    return base


def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test sets"""
    source = Path(source_dir)
    
    # Get all label directories
    label_dirs = [d for d in source.iterdir() if d.is_dir()]
    
    for label_dir in label_dirs:
        images = list(label_dir.glob("*.*"))
        random.shuffle(images)
        
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create split directories
        for split, split_images in [("train", train_images), 
                                     ("val", val_images), 
                                     ("test", test_images)]:
            split_dir = source.parent / split / label_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                dest = split_dir / img.name
                # Copy or move
                import shutil
                shutil.copy(img, dest)
        
        print(f"  {label_dir.name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    print("\n✅ Dataset split complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPI-600 Data Augmentation Tool")
    parser.add_argument("command", choices=["augment", "split", "init"],
                       help="Command to execute")
    parser.add_argument("--source", "-s", default="raw",
                       help="Source directory for images")
    parser.add_argument("--output", "-o", default="augmented",
                       help="Output directory")
    parser.add_argument("--labels", "-l", default="labels.json",
                       help="Labels JSON file")
    parser.add_argument("--multiplier", "-m", type=int, default=10,
                       help="Augmentation multiplier")
    
    args = parser.parse_args()
    
    if args.command == "init":
        create_dataset_structure()
    elif args.command == "augment":
        augmentor = DataAugmentor(output_dir=args.output)
        augmentor.augment_dataset(args.source, args.labels, args.multiplier)
    elif args.command == "split":
        split_dataset(args.source)
