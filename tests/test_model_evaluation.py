import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import ModelEvaluator
from src.train import ModelTrainer

class TestModelEvaluation:
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config for testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'data': {
                'processed_data_path': f"{temp_dir}/processed/",
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'n_estimators': 10,  # Small for fast testing
                'random_state': 42,
                'max_depth': 3
            },
            'output': {
                'model_path': f"{temp_dir}/models/",
                'metrics_path': f"{temp_dir}/models/metrics.json",
                'plots_path': f"{temp_dir}/models/plots/"
            },
            'training': {
                'cv_folds': 3,
                'scoring': 'accuracy'
            }
        }
        
        # Create directories
        os.makedirs(config['data']['processed_data_path'], exist_ok=True)
        os.makedirs(config['output']['model_path'], exist_ok=True)
        
        yield config, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model_and_data(self, temp_config):
        """Create sample model and test data"""
        config, temp_dir = temp_config
        
        # Generate sample data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Create and save model
        model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
        # model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and data
        joblib.dump(model, f"{config['output']['model_path']}/iris_model.pkl")
        pd.DataFrame(X_test_scaled).to_csv(f"{config['data']['processed_data_path']}/X_test.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{config['data']['processed_data_path']}/y_test.csv", index=False)
        
        return model, X_test_scaled, y_test, config
    
    def test_model_loading(self, sample_model_and_data, temp_config):
        """Test model and data loading"""
        model, X_test, y_test, config = sample_model_and_data
        
        # Write config to temporary file
        import yaml
        config_path = f"{temp_config[1]}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = ModelEvaluator(config_path)
        loaded_model, loaded_X_test, loaded_y_test = evaluator.load_model_and_data()
        
        assert loaded_model is not None
        assert isinstance(loaded_X_test, pd.DataFrame)
        assert len(loaded_y_test) > 0
    
    def test_model_evaluation_metrics(self, sample_model_and_data):
        """Test model evaluation returns correct metrics"""
        model, X_test, y_test, config = sample_model_and_data
        
        # Write config to temporary file
        import yaml
        config_path = f"{config['output']['model_path']}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = ModelEvaluator(config_path)
        X_test_df = pd.DataFrame(X_test)
        metrics, class_report, y_pred = evaluator.evaluate_model(model, X_test_df, y_test)
        
        # Check metrics exist and are valid
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        
        # Check predictions
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1, 2})  # Valid IRIS classes
    
    def test_metrics_saving(self, sample_model_and_data):
        """Test that metrics are saved correctly"""
        model, X_test, y_test, config = sample_model_and_data
        
        # Write config to temporary file
        import yaml
        config_path = f"{config['output']['model_path']}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = ModelEvaluator(config_path)
        X_test_df = pd.DataFrame(X_test)
        metrics, class_report, y_pred = evaluator.evaluate_model(model, X_test_df, y_test)
        evaluator.save_metrics(metrics, class_report)
        
        # Check if metrics file exists
        assert os.path.exists(config['output']['metrics_path'])
        
        # Check if metrics file contains correct data
        import json
        with open(config['output']['metrics_path'], 'r') as f:
            saved_data = json.load(f)
        
        assert 'metrics' in saved_data
        assert 'classification_report' in saved_data
        assert saved_data['metrics']['accuracy'] == metrics['accuracy']
    
    def test_model_performance_threshold(self, sample_model_and_data):
        """Test that model meets minimum performance threshold"""
        model, X_test, y_test, config = sample_model_and_data
        
        # Write config to temporary file
        import yaml
        config_path = f"{config['output']['model_path']}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = ModelEvaluator(config_path)
        X_test_df = pd.DataFrame(X_test)
        metrics, class_report, y_pred = evaluator.evaluate_model(model, X_test_df, y_test)
        
        # IRIS is an easy dataset, model should achieve high accuracy
        assert metrics['accuracy'] >= 0.8, f"Model accuracy {metrics['accuracy']:.4f} is below threshold"
        assert metrics['f1_score'] >= 0.8, f"Model F1-score {metrics['f1_score']:.4f} is below threshold"
