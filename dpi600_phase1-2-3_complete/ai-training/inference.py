"""
DPI-600 Drug Logo Inference
============================
ใช้ AI Model ที่ Train แล้วในการจำแนกภาพเม็ดยา
- รองรับ Single image และ Batch
- รองรับ REST API
- Export ผลลัพธ์เป็น JSON
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Import model class
from train_model import DrugLogoClassifier, CLASS_NAMES, CLASS_NAMES_TH, CONFIG


class DrugLogoPredictor:
    """Drug Logo Prediction Service"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model (.pth file)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Classes: {len(CLASS_NAMES)}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        config = checkpoint.get('config', CONFIG)
        model_name = config.get('model_name', 'efficientnet_b0')
        num_classes = config.get('num_classes', 10)
        
        # Create model
        model = DrugLogoClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self):
        """Get image transform for inference"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path, top_k=3):
        """
        Predict single image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            dict with predictions
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)
        
        # Format results
        predictions = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            predictions.append({
                'rank': i + 1,
                'label_id': CLASS_NAMES[idx],
                'label_th': CLASS_NAMES_TH.get(CLASS_NAMES[idx], ''),
                'label_en': CLASS_NAMES[idx].replace('_', ' ').title(),
                'confidence': round(prob * 100, 2)
            })
        
        result = {
            'image': str(image_path),
            'top_prediction': predictions[0] if predictions else None,
            'all_predictions': predictions,
            'predicted_at': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict multiple images.
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction dicts
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict_single(path, top_k)
                results.append(result)
            except Exception as e:
                results.append({
                    'image': str(path),
                    'error': str(e),
                    'predicted_at': datetime.now().isoformat()
                })
        
        return results
    
    def predict_directory(self, directory, top_k=3, output_file=None):
        """
        Predict all images in a directory.
        
        Args:
            directory: Directory containing images
            top_k: Number of top predictions per image
            output_file: Optional JSON file to save results
            
        Returns:
            List of prediction dicts
        """
        dir_path = Path(directory)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_paths = [f for f in dir_path.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_paths)} images in {directory}")
        
        results = []
        for i, path in enumerate(image_paths):
            result = self.predict_single(path, top_k)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images...")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


# =====================================================
# SIMPLE REST API (Flask)
# =====================================================
def create_api(model_path):
    """Create simple Flask API for inference"""
    try:
        from flask import Flask, request, jsonify
        from werkzeug.utils import secure_filename
        import tempfile
    except ImportError:
        print("Flask not installed. Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    predictor = DrugLogoPredictor(model_path)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'model': 'loaded'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            result = predictor.predict_single(tmp.name)
            os.unlink(tmp.name)
        
        return jsonify(result)
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                file.save(tmp.name)
                result = predictor.predict_single(tmp.name)
                results.append(result)
                os.unlink(tmp.name)
        
        return jsonify(results)
    
    return app


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="DPI-600 Drug Logo Inference")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--image", "-i", help="Single image to predict")
    parser.add_argument("--directory", "-d", help="Directory of images to predict")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Top K predictions")
    parser.add_argument("--api", action="store_true", help="Start REST API server")
    parser.add_argument("--port", type=int, default=5000, help="API port")
    
    args = parser.parse_args()
    
    if args.api:
        # Start API server
        app = create_api(args.model)
        if app:
            print(f"\n🚀 Starting API server on port {args.port}")
            print(f"   POST /predict - Single image prediction")
            print(f"   POST /predict/batch - Batch prediction")
            print(f"   GET /health - Health check\n")
            app.run(host='0.0.0.0', port=args.port, debug=False)
    else:
        # CLI prediction
        predictor = DrugLogoPredictor(args.model)
        
        if args.image:
            # Single image
            result = predictor.predict_single(args.image, args.top_k)
            print(f"\n📷 Image: {args.image}")
            print(f"🏷️  Prediction: {result['top_prediction']['label_th']} "
                  f"({result['top_prediction']['confidence']}%)")
            print(f"\nAll predictions:")
            for pred in result['all_predictions']:
                print(f"  {pred['rank']}. {pred['label_th']} ({pred['label_en']}): {pred['confidence']}%")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        elif args.directory:
            # Directory
            results = predictor.predict_directory(args.directory, args.top_k, args.output)
            
            # Summary
            print(f"\n📊 Summary:")
            label_counts = {}
            for r in results:
                if 'top_prediction' in r and r['top_prediction']:
                    label = r['top_prediction']['label_th']
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                print(f"  {label}: {count} images")
        
        else:
            print("Please provide --image or --directory")


if __name__ == "__main__":
    main()
