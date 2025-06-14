#!/usr/bin/env python3
"""
Deployment Script for Fraud Job Detection
Handles model training and deployment preparation
"""

import os
import sys
import argparse
import pickle
from main import FraudJobDetector
import pandas as pd

class ModelDeployment:
    def __init__(self):
        self.detector = FraudJobDetector()
        self.model_dir = "deployment_models"
        
    def prepare_deployment(self, train_data_path, test_data_path=None):
        """Prepare model for deployment"""
        print("🚀 Preparing model for deployment...")
        
        # Load and preprocess data
        train_data, test_data = self.detector.load_data(train_data_path, test_data_path)
        if train_data is None:
            raise Exception("Failed to load training data")
        
        # Preprocess
        train_data = self.detector.preprocess_data(train_data)
        
        # Balance data (undersample for faster training)
        train_data = self.detector.balance_data(train_data, method='undersample')
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            train_data['text'],
            train_data['fraudulent'],
            test_size=0.2,  # Smaller validation set for deployment
            stratify=train_data['fraudulent'],
            random_state=42
        )
        
        # Prepare features
        X_train_vec, _ = self.detector.prepare_features(pd.DataFrame({'text': X_train}))
        X_val_vec = self.detector.vectorizer.transform(X_val)
        
        # Train only XGBoost for deployment
        print("🔄 Training XGBoost model for deployment...")
        from imblearn.over_sampling import SMOTE
        from xgboost import XGBClassifier
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)
        
        # Train XGBoost with optimized parameters for deployment
        xgb_model = XGBClassifier(
            n_estimators=100,  # Reduced for faster inference
            max_depth=5,       # Reduced complexity
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        
        xgb_model.fit(X_resampled, y_resampled)
        
        # Evaluate
        y_pred = xgb_model.predict(X_val_vec)
        from sklearn.metrics import classification_report, f1_score
        
        f1 = f1_score(y_val, y_pred)
        print(f"✅ XGBoost Deployment Model - F1 Score: {f1:.4f}")
        
        # Store model
        self.detector.models['XGBoost'] = xgb_model
        
        # Save deployment artifacts
        self.save_deployment_artifacts()
        
        return f1
    
    def save_deployment_artifacts(self):
        """Save only essential artifacts for deployment"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save vectorizer
        with open(f'{self.model_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.detector.vectorizer, f)
        
        # Save XGBoost model
        with open(f'{self.model_dir}/xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.detector.models['XGBoost'], f)
        
        # Save model metadata
        metadata = {
            'model_type': 'XGBoost',
            'vectorizer_features': self.detector.vectorizer.max_features,
            'model_version': '1.0.0',
            'deployment_ready': True
        }
        
        with open(f'{self.model_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Deployment artifacts saved to {self.model_dir}/")
    
    def load_deployment_model(self):
        """Load deployment model for inference"""
        try:
            # Load vectorizer
            with open(f'{self.model_dir}/vectorizer.pkl', 'rb') as f:
                self.detector.vectorizer = pickle.load(f)
            
            # Load XGBoost model
            with open(f'{self.model_dir}/xgboost_model.pkl', 'rb') as f:
                self.detector.models['XGBoost'] = pickle.load(f)
            
            # Load metadata
            with open(f'{self.model_dir}/metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ Deployment model loaded successfully")
            print(f"   Model: {metadata['model_type']}")
            print(f"   Version: {metadata['model_version']}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load deployment model: {e}")
            return False
    
    def test_deployment(self, sample_texts):
        """Test deployment model with sample texts"""
        if not self.load_deployment_model():
            return False
        
        print("\n🧪 Testing deployment model...")
        
        predictions, probabilities = self.detector.predict_deployment(sample_texts)
        
        for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
            result = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
            print(f"Sample {i+1}: {result} (probability: {prob:.3f})")
            print(f"Text: {text[:100]}...")
            print("-" * 50)
        
        return True
    
    def create_deployment_package(self):
        """Create a complete deployment package"""
        print("📦 Creating deployment package...")
        
        # Create deployment directory structure
        package_dir = "fraud_detection_deployment"
        os.makedirs(package_dir, exist_ok=True)
        os.makedirs(f"{package_dir}/models", exist_ok=True)
        
        # Copy essential files
        import shutil
        
        # Copy model files
        if os.path.exists(self.model_dir):
            shutil.copytree(self.model_dir, f"{package_dir}/models", dirs_exist_ok=True)
        
        # Create deployment API
        api_code = '''#!/usr/bin/env python3
"""
Deployment API for Fraud Job Detection
Usage: python deployment_api.py
"""

import pickle
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

class FraudDetectionAPI:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open(f'{self.model_dir}/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(f'{self.model_dir}/xgboost_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict_single(self, text):
        """Predict fraud for a single job description"""
        if not self.model or not self.vectorizer:
            raise Exception("Model not loaded")
        
        # Preprocess and vectorize
        text_vector = self.vectorizer.transform([text.lower().strip()])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0][1]
        
        return {
            'prediction': int(prediction),
            'is_fraudulent': bool(prediction),
            'fraud_probability': float(probability),
            'timestamp': datetime.now().isoformat()
        }

# Initialize API
api = FraudDetectionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        result = api.predict_single(data['text'])
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        # Save API file
        with open(f"{package_dir}/deployment_api.py", 'w') as f:
            f.write(api_code)
        
        # Create requirements.txt
        requirements = """flask==2.3.2
scikit-learn==1.3.0
xgboost==1.7.6
pandas==2.0.3
numpy==1.24.3
imbalanced-learn==0.10.1
"""
        
        with open(f"{package_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
        
        # Create run script
        run_script = """#!/bin/bash
echo "Starting Fraud Detection API..."
python deployment_api.py
"""
        
        with open(f"{package_dir}/run.sh", 'w') as f:
            f.write(run_script)
        os.chmod(f"{package_dir}/run.sh", 0o755)
        
        # Create README
        readme = """# Fraud Detection Deployment

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Start API: `python deployment_api.py` or `./run.sh`
3. Test: `curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text":"Your job description here"}'`

## API Endpoints
- GET `/health` - Health check
- POST `/predict` - Predict fraud (JSON: {"text": "job description"})
"""
        
        with open(f"{package_dir}/README.md", 'w') as f:
            f.write(readme)
        
        print(f"✅ Deployment package created in {package_dir}/")
        return package_dir

def main():
    parser = argparse.ArgumentParser(description='Deploy Fraud Detection Model')
    parser.add_argument('--train-data', required=True, help='Path to training data')
    parser.add_argument('--test-data', help='Path to test data (optional)')
    parser.add_argument('--mode', choices=['prepare', 'test', 'package'], 
                       default='prepare', help='Deployment mode')
    
    args = parser.parse_args()
    
    deployment = ModelDeployment()
    
    if args.mode == 'prepare':
        # Prepare model for deployment
        f1_score = deployment.prepare_deployment(args.train_data, args.test_data)
        print(f"🎯 Model prepared with F1 Score: {f1_score:.4f}")
        
        # Create deployment package
        package_dir = deployment.create_deployment_package()
        print(f"📦 Deployment package ready at: {package_dir}")
        
    elif args.mode == 'test':
        # Test deployment with sample data
        sample_texts = [
            "Earn $5000 per week working from home! No experience required!",
            "Software Engineer position at Microsoft. 5+ years experience required.",
            "URGENT! Make money fast! Work from home! Easy money!",
            "Data Scientist role. PhD in Statistics preferred. Competitive salary."
        ]
        
        deployment.test_deployment(sample_texts)
        
    elif args.mode == 'package':
        # Just create the deployment package
        package_dir = deployment.create_deployment_package()
        print(f"📦 Deployment package created at: {package_dir}")

if __name__ == "__main__":
    main()