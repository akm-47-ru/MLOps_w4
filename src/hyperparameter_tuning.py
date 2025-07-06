import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import sys
import os
import logging
from itertools import product
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import DataLoader

class HyperparameterTuner:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set MLFlow tracking URI and experiment
        mlflow.set_tracking_uri(self.config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'iris_hyperparameter_tuning'))
    
    def generate_param_combinations(self):
        """Generate parameter combinations from config"""
        param_grid = self.config['hyperparameter_tuning']['param_grid']
        
        # Convert to sklearn ParameterGrid format
        sklearn_param_grid = {}
        for param, values in param_grid.items():
            sklearn_param_grid[param] = values
        
        return list(ParameterGrid(sklearn_param_grid))
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def run_hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        """Run hyperparameter tuning with MLFlow tracking"""
        param_combinations = self.generate_param_combinations()
        best_score = -np.inf
        best_params = None
        best_model = None
        
        self.logger.info(f"Starting hyperparameter tuning with {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            with mlflow.start_run(run_name=f"run_{i+1}"):
                # Log parameters
                mlflow.log_params(params)
                
                # Create and train model
                model = RandomForestClassifier(
                    random_state=self.config['model']['random_state'],
                    **params
                )
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.config['training']['cv_folds'],
                    scoring=self.config['training']['scoring']
                )
                
                # Train final model for evaluation
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test, y_test)
                
                # Log metrics
                mlflow.log_metric("cv_mean_score", cv_scores.mean())
                mlflow.log_metric("cv_std_score", cv_scores.std())
                
                for metric_name, metric_value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Check if this is the best model
                current_score = cv_scores.mean()
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                    best_model = model
                
                self.logger.info(f"Run {i+1}/{len(param_combinations)}: CV Score = {current_score:.4f}, Test Accuracy = {test_metrics['accuracy']:.4f}")
        
        return best_model, best_params, best_score
    
    def save_best_model(self, model, scaler, params):
        """Save the best model and parameters"""
        model_dir = self.config['output']['model_path']
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(model, f"{model_dir}/best_iris_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
        
        # Save best parameters
        with open(f"{model_dir}/best_params.yaml", 'w') as f:
            yaml.dump(params, f)
        
        self.logger.info(f"Best model saved to {model_dir}")

def main():
    # Load and preprocess data
    data_loader = DataLoader()
    df = data_loader.load_raw_data()
    data_loader.validate_data(df)
    X_train, X_test, y_train, y_test, scaler = data_loader.preprocess_data(df)
    
    # Run hyperparameter tuning
    tuner = HyperparameterTuner()
    best_model, best_params, best_score = tuner.run_hyperparameter_tuning(
        X_train, X_test, y_train, y_test
    )
    
    # Save best model
    tuner.save_best_model(best_model, scaler, best_params)
    
    print(f"Hyperparameter tuning completed!")
    print(f"Best CV Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    main()