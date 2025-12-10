"""
DPI-600 Drug Logo Recognition Model
====================================
Train CNN model for drug pill logo classification
- Transfer Learning with EfficientNet/ResNet
- Support for PyTorch
- Azure ML integration ready
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================
CONFIG = {
    'model_name': 'efficientnet_b0',  # or 'resnet50', 'mobilenet_v3'
    'num_classes': 10,  # Number of drug logo classes
    'image_size': 224,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Class names mapping
CLASS_NAMES = [
    'lion', 'wy', '999', 'horse', 'r_mark',
    'star', 'eagle', 'no_logo', 'ice', 'heroin'
]

CLASS_NAMES_TH = {
    'lion': 'สิงโต',
    'wy': 'WY',
    '999': '999',
    'horse': 'ม้า',
    'r_mark': 'R',
    'star': 'ดาว',
    'eagle': 'นกอินทรี',
    'no_logo': 'ไม่มีโลโก้',
    'ice': 'ไอซ์',
    'heroin': 'เฮโรอีน'
}


# =====================================================
# DATASET
# =====================================================
class DrugPillDataset(Dataset):
    """Custom Dataset for drug pill images"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        # Load all images
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size, is_training=True):
    """Get image transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# =====================================================
# MODEL
# =====================================================
class DrugLogoClassifier(nn.Module):
    """Drug Logo Classification Model using Transfer Learning"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=10, pretrained=True):
        super(DrugLogoClassifier, self).__init__()
        
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        
        elif model_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone for fine-tuning only classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        if self.model_name == 'efficientnet_b0':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == 'resnet50':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'mobilenet_v3':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# =====================================================
# TRAINING
# =====================================================
class Trainer:
    """Model trainer with early stopping and checkpointing"""
    
    def __init__(self, model, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0
        self.patience_counter = 0
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs, lr, patience=5):
        """Full training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Epochs: {epochs}, LR: {lr}, Patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
            'class_names': CLASS_NAMES,
            'class_names_th': CLASS_NAMES_TH,
            'config': CONFIG
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
    
    def evaluate(self, test_loader):
        """Evaluate on test set with detailed metrics"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        accuracy = (all_preds == all_labels).mean() * 100
        
        # Per-class accuracy
        class_accuracy = {}
        for idx, class_name in enumerate(CLASS_NAMES):
            mask = all_labels == idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
                class_accuracy[class_name] = class_acc
        
        # Confusion matrix
        num_classes = len(CLASS_NAMES)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion_matrix[label, pred] += 1
        
        results = {
            'overall_accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion_matrix.tolist(),
            'num_samples': len(all_labels)
        }
        
        return results


# =====================================================
# EXPORT FOR INFERENCE
# =====================================================
def export_model(model, save_path, input_size=(1, 3, 224, 224)):
    """Export model for inference"""
    model.eval()
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'class_names_th': CLASS_NAMES_TH
    }, f"{save_path}.pth")
    
    # Export to ONNX
    dummy_input = torch.randn(input_size)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to {save_path}.pth and {save_path}.onnx")


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="DPI-600 Drug Logo Recognition Training")
    parser.add_argument("--data-dir", "-d", required=True, help="Dataset directory")
    parser.add_argument("--model", "-m", default="efficientnet_b0",
                       choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                       help="Model architecture")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--freeze-backbone", action="store_true", 
                       help="Freeze backbone (fine-tune classifier only)")
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['model_name'] = args.model
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.lr
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Data transforms
    train_transform = get_transforms(CONFIG['image_size'], is_training=True)
    val_transform = get_transforms(CONFIG['image_size'], is_training=False)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DrugPillDataset(args.data_dir, 'train', train_transform)
    val_dataset = DrugPillDataset(args.data_dir, 'val', val_transform)
    test_dataset = DrugPillDataset(args.data_dir, 'test', val_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = DrugLogoClassifier(
        model_name=args.model,
        num_classes=CONFIG['num_classes'],
        pretrained=True
    )
    
    if args.freeze_backbone:
        print("Freezing backbone layers...")
        model.freeze_backbone()
    
    # Create trainer
    trainer = Trainer(model, CONFIG['device'], save_dir=output_dir / 'checkpoints')
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        patience=CONFIG['early_stopping_patience']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    print(f"\n📊 Test Results:")
    print(f"   Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"   Per-class Accuracy:")
    for class_name, acc in results['class_accuracy'].items():
        print(f"     {class_name} ({CLASS_NAMES_TH.get(class_name, '')}): {acc:.2f}%")
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'results': {
                'overall_accuracy': results['overall_accuracy'],
                'class_accuracy': results['class_accuracy'],
                'num_samples': results['num_samples']
            },
            'history': history,
            'evaluated_at': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    # Export best model
    print("\nExporting model...")
    best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    export_model(model, str(output_dir / 'dpi600_drug_logo_model'))
    
    print(f"\n✅ Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
