import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

class ModelEvaluator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_data(self):
        """Load trained model and test data"""
        model_path = f"{self.config['output']['model_path']}/iris_model.pkl"
        processed_dir = self.config['data']['processed_data_path']
        
        model = joblib.load(model_path)
        X_test = pd.read_csv(f"{processed_dir}/X_test.csv")
        y_test = pd.read_csv(f"{processed_dir}/y_test.csv").squeeze()
        
        return model, X_test, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        self.logger.info(f"Model Evaluation Metrics: {metrics}")
        
        return metrics, class_report, y_pred
    
    def save_metrics(self, metrics, class_report):
        """Save evaluation metrics"""
        output_data = {
            'metrics': metrics,
            'classification_report': class_report
        }
        
        metrics_path = self.config['output']['metrics_path']
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def create_plots(self, y_test, y_pred):
        """Create evaluation plots"""
        plots_dir = self.config['output']['plots_path']
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{plots_dir}/confusion_matrix.png")
        plt.close()
        
        self.logger.info(f"Plots saved to {plots_dir}")

def main():
    evaluator = ModelEvaluator()
    model, X_test, y_test = evaluator.load_model_and_data()
    metrics, class_report, y_pred = evaluator.evaluate_model(model, X_test, y_test)
    evaluator.save_metrics(metrics, class_report)
    evaluator.create_plots(y_test, y_pred)
    
    print("Evaluation completed successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
