import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import yaml
import os
import logging
from src.data_loader import DataLoader

# Training module
class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        model = RandomForestClassifier(
            n_estimators=self.config['model']['n_estimators'],
            random_state=self.config['model']['random_state'],
            max_depth=self.config['model']['max_depth']
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring']
        )
        
        # Train final model
        model.fit(X_train, y_train)
        
        self.logger.info(f"Model trained with CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, cv_scores
    
    def save_model(self, model, scaler):
        """Save the trained model and scaler"""
        model_dir = self.config['output']['model_path']
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, f"{model_dir}/iris_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
        
        self.logger.info(f"Model and scaler saved to {model_dir}")

def main():
    # Load and preprocess data
    data_loader = DataLoader()
    df = data_loader.load_raw_data()
    data_loader.validate_data(df)
    X_train, X_test, y_train, y_test, scaler = data_loader.preprocess_data(df)
    
    # Train model
    trainer = ModelTrainer()
    model, cv_scores = trainer.train_model(X_train, y_train)
    trainer.save_model(model, scaler)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
