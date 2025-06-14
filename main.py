#!/usr/bin/env python3
"""
Main ML Pipeline for Fraud Job Detection
Handles data preprocessing, model training, and prediction
Enhanced with debug print statements
"""
import time
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

class FraudJobDetector:
    def __init__(self):
        print("🔧 Initializing FraudJobDetector class...")
        self.models = {}
        self.vectorizer = None
        self.stop_words = None
        self._download_nltk_data()
        print("✅ FraudJobDetector initialized successfully!")
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        print("📦 Checking NLTK data...")
        try:
            nltk.data.find('corpora/stopwords')
            print("✅ NLTK stopwords already available")
        except LookupError:
            print("⬇️ Downloading NLTK stopwords...")
            nltk.download('stopwords')
            print("✅ NLTK stopwords downloaded")
        self.stop_words = set(stopwords.words('english'))
        print(f"📝 Loaded {len(self.stop_words)} stopwords")
    
    def load_data(self, train_path="Fraud_data.csv", test_path="Fraud_test.csv"):
        """Load and return training and test datasets"""
        print(f"📂 Loading training data from: {train_path}")
        print(f"📂 Loading test data from: {test_path}")
        
        try:
            print("⏳ Reading training data...")
            train_data = pd.read_csv(train_path)
            print(f"✅ Training data loaded: {train_data.shape}")
            
            print("⏳ Reading test data...")
            test_data = pd.read_csv(test_path)
            print(f"✅ Test data loaded: {test_data.shape}")
            
            print(f"📊 Training data columns: {list(train_data.columns)}")
            if 'fraudulent' in train_data.columns:
                fraud_counts = train_data['fraudulent'].value_counts()
                print(f"📊 Fraud distribution in training: {fraud_counts.to_dict()}")
            
            return train_data, test_data
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("📁 Current directory files:")
            for file in os.listdir('.'):
                if file.endswith('.csv'):
                    print(f"   - {file}")
            return None, None
        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            return None, None
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the dataset"""
        print(f"🔄 Preprocessing {'training' if is_training else 'test'} data...")
        print(f"   Original shape: {df.shape}")
        
        # Drop unnecessary columns
        cols_to_drop = ['job_id', 'salary_range', 'telecommuting', 
                       'has_company_logo', 'has_questions']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            print(f"🗑️ Dropping columns: {existing_cols_to_drop}")
            df = df.drop(existing_cols_to_drop, axis=1, errors='ignore')
        
        # Fill NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"🔧 Filling {nan_count} NaN values...")
            df = df.fillna(' ')
        
        # Create combined text column
        text_fields = ['title', 'location', 'company_profile', 'description',
                      'requirements', 'benefits', 'industry']
        existing_fields = [field for field in text_fields if field in df.columns]
        print(f"📝 Combining text fields: {existing_fields}")
        
        df['text'] = df[existing_fields].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1
        )
        
        # Convert to lowercase and remove stopwords
        print("🔤 Converting to lowercase...")
        df['text'] = df['text'].str.lower()
        
        print("🚫 Removing stopwords...")
        df['text'] = df['text'].apply(
            lambda x: ' '.join([word for word in x.split() 
                              if word not in self.stop_words])
        )
        
        # Extract country from location for analysis
        if 'location' in df.columns:
            print("🌍 Extracting country information...")
            df['country'] = df['location'].apply(self._extract_country)
        
        print(f"✅ Preprocessing complete. Final shape: {df.shape}")
        print(f"📏 Average text length: {df['text'].str.len().mean():.0f} characters")
        
        return df
    
    def _extract_country(self, location):
        """Extract country from location string"""
        try:
            return location.split(',')[0].strip()
        except:
            return "Unknown"
    
    def balance_data(self, df, method='undersample'):
        """Balance the dataset using specified method"""
        print(f"⚖️ Balancing data using {method} method...")
        
        if 'fraudulent' not in df.columns:
            print("⚠️ No 'fraudulent' column found, skipping balancing")
            return df
            
        X = df.drop('fraudulent', axis=1)
        y = df['fraudulent']
        
        print(f"📊 Original class distribution: {y.value_counts().to_dict()}")
        
        if method == 'undersample':
            print("🔻 Applying random undersampling...")
            sampler = RandomUnderSampler(random_state=42)
        else:  # oversample will be handled during model training
            print("ℹ️ Oversampling will be handled during training")
            return df
            
        X_res, y_res = sampler.fit_resample(X, y)
        balanced_df = pd.concat([X_res, y_res], axis=1)
        
        print(f"✅ Data balanced:")
        print(f"   Original shape: {df.shape}")
        print(f"   Balanced shape: {balanced_df.shape}")
        print(f"   New distribution: {balanced_df['fraudulent'].value_counts().to_dict()}")
        
        return balanced_df
    
    def prepare_features(self, train_df, test_df=None):
        """Prepare features for model training"""
        print("🔤 Preparing TF-IDF features...")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        print("   TF-IDF parameters: max_features=5000, stop_words='english'")
        
        # Fit and transform training data
        print("⏳ Fitting TF-IDF on training data...")
        X_train_vec = self.vectorizer.fit_transform(train_df['text'])
        print(f"✅ Training features shape: {X_train_vec.shape}")
        print(f"📚 Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Transform test data if provided
        X_test_vec = None
        if test_df is not None:
            print("⏳ Transforming test data...")
            X_test_vec = self.vectorizer.transform(test_df['text'])
            print(f"✅ Test features shape: {X_test_vec.shape}")
        
        return X_train_vec, X_test_vec
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and return performance metrics"""
        print("🎯 Starting model training phase...")
        results = {}
        
        # Apply SMOTE for oversampling
        print("📈 Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"   Original training size: {X_train.shape[0]}")
        print(f"   After SMOTE: {X_resampled.shape[0]}")
        
        # Model configurations
        model_configs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        }
        
        print(f"🤖 Training {len(model_configs)} models...")
        
        # Train and evaluate each model
        for i, (name, model) in enumerate(model_configs.items(), 1):
            print(f"\n[{i}/{len(model_configs)}] 🔄 Training {name}...")
            
            # Train model
            print(f"   ⏳ Fitting {name} model...")
            model.fit(X_resampled, y_resampled)
            print(f"   ✅ {name} training completed")
            
            # Predict
            print(f"   🎯 Making predictions...")
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            print(f"   📊 Calculating metrics...")
            cm = confusion_matrix(y_val, y_pred)
            cr = classification_report(y_val, y_pred, output_dict=True)
            f1 = f1_score(y_val, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'confusion_matrix': cm,
                'classification_report': cr,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            # Store model
            self.models[name] = model
            
            print(f"   ✅ {name} completed - F1 Score: {f1:.4f}")
            print(f"   📈 Accuracy: {cr['accuracy']:.4f}")
            print(f"   🎯 Precision: {cr['weighted avg']['precision']:.4f}")
            print(f"   🔍 Recall: {cr['weighted avg']['recall']:.4f}")
        
        print(f"\n🎉 All {len(model_configs)} models trained successfully!")
        return results
    
    def predict(self, text_data, model_name='XGBoost'):
        """Make predictions on new data"""
        print(f"🔮 Making predictions using {model_name}...")
        
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please train models first.")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Vectorize input
        print("   🔤 Vectorizing input text...")
        X_vec = self.vectorizer.transform(text_data)
        print(f"   📊 Input features shape: {X_vec.shape}")
        
        # Predict
        print("   🎯 Generating predictions...")
        predictions = self.models[model_name].predict(X_vec)
        probabilities = self.models[model_name].predict_proba(X_vec)[:, 1] if hasattr(self.models[model_name], 'predict_proba') else None
        
        fraud_count = sum(predictions)
        print(f"   📊 Predictions: {fraud_count} fraudulent out of {len(predictions)} total")
        
        return predictions, probabilities
    
    def save_models(self, filepath='models', deploy_model_only=True):
        """Save trained models and vectorizer"""
        print(f"💾 Saving models to {filepath}/...")
        os.makedirs(filepath, exist_ok=True)
        
        # Save vectorizer
        print("   📝 Saving vectorizer...")
        with open(f'{filepath}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        if deploy_model_only:
            # Save only XGBoost for deployment
            if 'XGBoost' in self.models:
                print("   🚀 Saving XGBoost deployment model...")
                with open(f'{filepath}/xgboost_model.pkl', 'wb') as f:
                    pickle.dump(self.models['XGBoost'], f)
                print(f"✅ XGBoost model (deployment) saved to {filepath}/")
            else:
                print("⚠️ XGBoost model not found for deployment")
        else:
            # Save all models for comparison
            print(f"   💾 Saving all {len(self.models)} models...")
            for name, model in self.models.items():
                model_filename = f"{filepath}/{name.lower().replace(' ', '_')}_model.pkl"
                print(f"     - Saving {name}...")
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)
            print(f"✅ All models saved to {filepath}/")
    
    def load_models(self, filepath='models', deploy_only=True):
        """Load trained models and vectorizer"""
        print(f"📂 Loading models from {filepath}/...")
        
        try:
            # Load vectorizer
            print("   📝 Loading vectorizer...")
            with open(f'{filepath}/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("   ✅ Vectorizer loaded")
            
            if deploy_only:
                # Load only XGBoost for deployment
                try:
                    print("   🚀 Loading XGBoost deployment model...")
                    with open(f'{filepath}/xgboost_model.pkl', 'rb') as f:
                        self.models['XGBoost'] = pickle.load(f)
                    print(f"✅ XGBoost model (deployment) loaded from {filepath}/")
                    return True
                except FileNotFoundError:
                    print(f"❌ XGBoost deployment model not found in {filepath}/")
                    return False
            else:
                # Load all available models
                model_files = {
                    'Logistic Regression': 'logistic_regression_model.pkl',
                    'Naive Bayes': 'naive_bayes_model.pkl',
                    'XGBoost': 'xgboost_model.pkl'
                }
                
                loaded_count = 0
                for name, filename in model_files.items():
                    try:
                        print(f"   📦 Loading {name}...")
                        with open(f'{filepath}/{filename}', 'rb') as f:
                            self.models[name] = pickle.load(f)
                        loaded_count += 1
                        print(f"   ✅ {name} loaded")
                    except FileNotFoundError:
                        print(f"   ⚠️ Model file {filename} not found")
                
                print(f"✅ {loaded_count} models loaded from {filepath}/")
                return loaded_count > 0
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_deployment_model(self):
        """Get the XGBoost model for deployment"""
        if 'XGBoost' in self.models:
            return self.models['XGBoost']
        else:
            raise ValueError("XGBoost deployment model not available. Please train models first.")
    
    def predict_deployment(self, text_data):
        """Make predictions using the deployment model (XGBoost)"""
        print("🚀 Making deployment predictions...")
        
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please train models first.")
        
        if 'XGBoost' not in self.models:
            raise ValueError("XGBoost deployment model not found. Please train models first.")
        
        # Vectorize input
        print("   🔤 Vectorizing input...")
        X_vec = self.vectorizer.transform(text_data)
        
        # Predict using XGBoost
        print("   🎯 Generating predictions...")
        model = self.models['XGBoost']
        predictions = model.predict(X_vec)
        probabilities = model.predict_proba(X_vec)[:, 1]
        
        fraud_count = sum(predictions)
        print(f"   📊 Deployment predictions: {fraud_count} fraudulent out of {len(predictions)} total")
        
        return predictions, probabilities

def main():
    """Main execution function"""
    print("🚀 Fraud Job Detection Pipeline")
    print("="*50)
    print("⏰ Starting execution...")
    
    # Initialize detector
    print("\n📋 Step 1: Initializing detector...")
    detector = FraudJobDetector()
    
    # Load data
    print("\n📋 Step 2: Loading data...")
    train_data, test_data = detector.load_data()
    if train_data is None:
        print("❌ Failed to load data. Please check file paths.")
        print("🔍 Make sure 'Fraud_data.csv' and 'Fraud_test.csv' exist in current directory")
        return
    
    # Preprocess data
    print("\n📋 Step 3: Preprocessing data...")
    train_data = detector.preprocess_data(train_data, is_training=True)
    test_data = detector.preprocess_data(test_data, is_training=False)
    
    # Balance training data
    print("\n📋 Step 4: Balancing training data...")
    train_data = detector.balance_data(train_data, method='undersample')
    
    # Split training data
    print("\n📋 Step 5: Splitting training data...")
    print("🔄 Creating train/validation split (70/30)...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_data['text'],
        train_data['fraudulent'],
        test_size=0.3,
        stratify=train_data['fraudulent'],
        random_state=42
    )
    print(f"✅ Split complete:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Prepare features
    print("\n📋 Step 6: Preparing features...")
    X_train_vec, X_test_vec = detector.prepare_features(
        pd.DataFrame({'text': X_train}),
        pd.DataFrame({'text': test_data['text']})
    )
    print("🔄 Vectorizing validation data...")
    X_val_vec = detector.vectorizer.transform(X_val)
    print(f"✅ Validation features shape: {X_val_vec.shape}")
    
    # Train models
    print("\n📋 Step 7: Training models...")
    results = detector.train_models(X_train_vec, y_train, X_val_vec, y_val)
    
    # Display results
    print("\n📋 Step 8: Displaying results...")
    print("\n📊 Model Performance Summary:")
    print("="*50)
    for name, result in results.items():
        print(f"🏆 {name}: F1-Score = {result['f1_score']:.4f}")
    
    # Save models (deployment version - XGBoost only)
    print("\n📋 Step 9: Saving models...")
    detector.save_models(deploy_model_only=True)
    
    # Also save comparison version with all models
    print("💾 Saving comparison models...")
    detector.save_models(filepath='models_comparison', deploy_model_only=False)
    
    # Make test predictions using deployment model
    print("\n📋 Step 10: Making test predictions...")
    test_predictions, test_probs = detector.predict_deployment(test_data['text'])
    test_data['fraud_prediction'] = test_predictions
    if test_probs is not None:
        test_data['fraud_probability'] = test_probs
    
    # Save test results
    print("💾 Saving test results...")
    test_data.to_csv('test_predictions.csv', index=False)
    print("✅ Test predictions saved to test_predictions.csv")
    
    print("\n🎉 Pipeline completed successfully!")
    print("="*50)
    print("📁 Files created:")
    print("   - models/vectorizer.pkl")
    print("   - models/xgboost_model.pkl")
    print("   - models_comparison/ (all models)")
    print("   - test_predictions.csv")
    print("\n✨ Ready for deployment!")

if __name__ == "__main__":
    print("🏁 Script started!")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📋 Python version: {os.sys.version}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"\n💥 Script failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Script execution finished!")