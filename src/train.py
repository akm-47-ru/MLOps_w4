import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import yaml
import sys
import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader

# This is the function for training the model
class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLFlow
        mlflow.set_tracking_uri(self.config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'iris_model_training'))


def train_model(self, X_train, y_train):
    """Train the model with MLFlow tracking"""
    with mlflow.start_run(run_name="baseline_training"):
        # Log parameters
        model_params = {
            'n_estimators': self.config['model']['n_estimators'],
            'random_state': self.config['model']['random_state'],
            'max_depth': self.config['model']['max_depth']
        }
        mlflow.log_params(model_params)
        
        model = RandomForestClassifier(**model_params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring']
        )
        
        # Log cross-validation metrics
        mlflow.log_metric("cv_mean_score", cv_scores.mean())
        mlflow.log_metric("cv_std_score", cv_scores.std())
        
        # Train final model
        model.fit(X_train, y_train)
        
        input_example = X_train[:5]  
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  
            input_example=input_example,
            signature=signature
        )
        
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