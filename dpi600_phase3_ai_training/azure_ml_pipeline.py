"""
DPI-600 Azure ML Training Pipeline
===================================
Training pipeline for Azure Machine Learning
- Supports Azure ML Workspace
- Auto-scaling compute clusters
- Experiment tracking with MLflow
- Model registration and deployment
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# =====================================================
# AZURE ML CONFIGURATION
# =====================================================
AZURE_CONFIG = {
    'subscription_id': os.environ.get('AZURE_SUBSCRIPTION_ID', ''),
    'resource_group': os.environ.get('AZURE_RESOURCE_GROUP', 'dpi600-rg'),
    'workspace_name': os.environ.get('AZURE_WORKSPACE', 'dpi600-mlworkspace'),
    'compute_name': 'gpu-cluster',
    'compute_size': 'Standard_NC6s_v3',  # NVIDIA V100 GPU
    'min_nodes': 0,
    'max_nodes': 4,
    'experiment_name': 'dpi600-drug-logo-recognition',
    'environment_name': 'dpi600-pytorch-env',
}


def create_azure_ml_config():
    """Create Azure ML workspace configuration file"""
    config = {
        'subscription_id': AZURE_CONFIG['subscription_id'],
        'resource_group': AZURE_CONFIG['resource_group'],
        'workspace_name': AZURE_CONFIG['workspace_name']
    }
    
    config_dir = Path('.azureml')
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Azure ML config saved to {config_dir / 'config.json'}")
    return config


def setup_azure_ml_environment():
    """
    Setup Azure ML environment with required packages.
    Returns conda environment specification.
    """
    conda_env = {
        'name': AZURE_CONFIG['environment_name'],
        'channels': ['pytorch', 'nvidia', 'conda-forge', 'defaults'],
        'dependencies': [
            'python=3.10',
            'pytorch=2.0.1',
            'torchvision=0.15.2',
            'pytorch-cuda=11.8',
            'numpy=1.24.3',
            'pillow=10.0.0',
            'scikit-learn=1.3.0',
            'matplotlib=3.7.2',
            'pandas=2.0.3',
            {
                'pip': [
                    'azureml-core==1.53.0',
                    'azureml-mlflow==1.53.0',
                    'mlflow==2.6.0',
                    'onnx==1.14.0',
                    'onnxruntime-gpu==1.15.1',
                ]
            }
        ]
    }
    
    return conda_env


def create_training_script():
    """Create Azure ML compatible training script"""
    script = '''"""
DPI-600 Azure ML Training Script
================================
This script is executed on Azure ML compute cluster.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch

# =====================================================
# CONFIGURATION
# =====================================================
CLASS_NAMES = [
    'lion', 'wy', '999', 'horse', 'r_mark',
    'star', 'eagle', 'no_logo', 'ice', 'heroin'
]

CLASS_NAMES_TH = {
    'lion': 'สิงโต', 'wy': 'WY', '999': '999', 'horse': 'ม้า',
    'r_mark': 'R', 'star': 'ดาว', 'eagle': 'นกอินทรี',
    'no_logo': 'ไม่มีโลโก้', 'ice': 'ไอซ์', 'heroin': 'เฮโรอีน'
}


class DrugPillDataset(Dataset):
    """Drug pill image dataset"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DrugLogoClassifier(nn.Module):
    """EfficientNet-based classifier"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train(args):
    """Main training function"""
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model': 'efficientnet_b0',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'image_size': 224
        })
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {device}")
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = DrugPillDataset(args.data_dir, 'train', train_transform)
        val_dataset = DrugPillDataset(args.data_dir, 'val', val_transform)
        test_dataset = DrugPillDataset(args.data_dir, 'test', val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Model
        model = DrugLogoClassifier(num_classes=len(CLASS_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        best_val_acc = 0
        
        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Validate
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_acc)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'class_names': CLASS_NAMES,
                    'class_names_th': CLASS_NAMES_TH
                }, args.output_dir / 'best_model.pth')
                print(f"  ✓ New best model saved!")
        
        # Test evaluation
        model.load_state_dict(torch.load(args.output_dir / 'best_model.pth')['model_state_dict'])
        model.eval()
        
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        mlflow.log_metric('test_acc', test_acc)
        
        print(f"\\n🎯 Test Accuracy: {test_acc:.2f}%")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Export ONNX
        dummy = torch.randn(1, 3, 224, 224).to(device)
        onnx_path = args.output_dir / 'model.onnx'
        torch.onnx.export(model, dummy, onnx_path, opset_version=11)
        mlflow.log_artifact(str(onnx_path))
        
        print(f"✅ Training complete! Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('./outputs'))
    parser.add_argument('--experiment-name', default='dpi600-training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    
    train(args)
'''
    return script


def create_azure_ml_pipeline():
    """Create Azure ML pipeline configuration"""
    pipeline_config = '''# DPI-600 Azure ML Pipeline Configuration
# =====================================

name: dpi600-training-pipeline
description: Drug Logo Recognition Training Pipeline

# Compute target
compute:
  target: gpu-cluster
  instance_type: Standard_NC6s_v3
  min_instances: 0
  max_instances: 4

# Environment
environment:
  name: dpi600-pytorch-env
  docker:
    image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04
  conda:
    dependencies:
      - python=3.10
      - pytorch=2.0.1
      - torchvision=0.15.2
      - pip:
        - azureml-core
        - mlflow
        - onnx

# Pipeline steps
steps:
  - name: train
    script: train_azure.py
    arguments:
      - --data-dir
      - ${{inputs.dataset}}
      - --output-dir
      - ${{outputs.model}}
      - --epochs
      - ${{inputs.epochs}}
      - --batch-size
      - ${{inputs.batch_size}}
      - --lr
      - ${{inputs.learning_rate}}

# Inputs
inputs:
  dataset:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/dpi600/dataset
  epochs:
    type: integer
    default: 50
  batch_size:
    type: integer
    default: 32
  learning_rate:
    type: number
    default: 0.001

# Outputs
outputs:
  model:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/dpi600/models
'''
    return pipeline_config


def create_submission_script():
    """Create script to submit training job to Azure ML"""
    script = '''#!/usr/bin/env python3
"""
Submit DPI-600 Training Job to Azure ML
=======================================
Usage: python submit_job.py --data-path ./dataset
"""

import os
import argparse
from pathlib import Path
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Local dataset path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Connect to Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential)
    
    print(f"Connected to workspace: {ml_client.workspace_name}")
    
    # Upload dataset
    print("Uploading dataset...")
    data_asset = ml_client.data.create_or_update(
        Data(
            name="dpi600-drug-dataset",
            path=args.data_path,
            type="uri_folder"
        )
    )
    
    # Create job
    job = command(
        code="./",
        command="python train_azure.py --data-dir ${{inputs.data}} --output-dir ${{outputs.model}} --epochs ${{inputs.epochs}} --batch-size ${{inputs.batch_size}} --lr ${{inputs.lr}}",
        inputs={
            "data": Input(type="uri_folder", path=data_asset.path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr
        },
        outputs={
            "model": Output(type="uri_folder")
        },
        environment="dpi600-pytorch-env:1",
        compute="gpu-cluster",
        display_name="dpi600-training",
        experiment_name="dpi600-drug-logo-recognition"
    )
    
    # Submit
    print("Submitting job...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")
    
    # Wait for completion
    ml_client.jobs.stream(returned_job.name)
    
    print("\\n✅ Training complete!")

if __name__ == "__main__":
    main()
'''
    return script


# Save all files
def save_azure_ml_files(output_dir):
    """Save all Azure ML related files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training script
    with open(output_dir / 'train_azure.py', 'w') as f:
        f.write(create_training_script())
    
    # Pipeline config
    with open(output_dir / 'pipeline.yml', 'w') as f:
        f.write(create_azure_ml_pipeline())
    
    # Submission script
    with open(output_dir / 'submit_job.py', 'w') as f:
        f.write(create_submission_script())
    
    # Conda environment
    import yaml
    conda_env = setup_azure_ml_environment()
    with open(output_dir / 'conda_env.yml', 'w') as f:
        yaml.dump(conda_env, f, default_flow_style=False)
    
    print(f"✅ Azure ML files saved to {output_dir}")
    print(f"   - train_azure.py")
    print(f"   - pipeline.yml")
    print(f"   - submit_job.py")
    print(f"   - conda_env.yml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Azure ML for DPI-600")
    parser.add_argument("--output", "-o", default="azure_ml", help="Output directory")
    parser.add_argument("--create-config", action="store_true", help="Create workspace config")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_azure_ml_config()
    
    save_azure_ml_files(args.output)
