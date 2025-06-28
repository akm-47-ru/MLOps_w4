import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader

class TestDataValidation:
    
    @pytest.fixture
    def data_loader(self):
        return DataLoader()
    
    @pytest.fixture
    def sample_valid_data(self):
        """Create sample valid IRIS data"""
        np.random.seed(42)
        data = {
            'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
            'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
            'petal length (cm)': np.random.normal(3.8, 1.8, 150),
            'petal width (cm)': np.random.normal(1.2, 0.7, 150),
            'target': np.repeat([0, 1, 2], 50),
            'target_name': np.repeat(['setosa', 'versicolor', 'virginica'], 50)
        }
        return pd.DataFrame(data)
    
    def test_load_raw_data(self, data_loader):
        """Test raw data loading"""
        df = data_loader.load_raw_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150
        assert 'target' in df.columns
        assert 'target_name' in df.columns
    
    def test_data_validation_valid_data(self, data_loader, sample_valid_data):
        """Test data validation with valid data"""
        assert data_loader.validate_data(sample_valid_data) == True
    
    def test_data_validation_missing_values(self, data_loader, sample_valid_data):
        """Test data validation with missing values"""
        sample_valid_data.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="Dataset contains missing values"):
            data_loader.validate_data(sample_valid_data)
    
    def test_data_validation_wrong_target_values(self, data_loader, sample_valid_data):
        """Test data validation with wrong target values"""
        sample_valid_data['target'] = [0, 1, 2, 3] * 37 + [0, 1]  # Add invalid target 3
        with pytest.raises(ValueError, match="Target values are not as expected"):
            data_loader.validate_data(sample_valid_data)
    
    def test_data_validation_wrong_shape(self, data_loader):
        """Test data validation with wrong data shape"""
        wrong_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'target': [0, 1, 2]
        })
        with pytest.raises(ValueError, match="Dataset shape is not as expected"):
            data_loader.validate_data(wrong_data)
    
    def test_preprocess_data(self, data_loader, sample_valid_data):
        """Test data preprocessing"""
        X_train, X_test, y_train, y_test, scaler = data_loader.preprocess_data(sample_valid_data)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert len(X_train) + len(X_test) == 150
        assert len(y_train) + len(y_test) == 150
        assert X_train.shape[1] == 4  # 4 features
        assert scaler is not None
    
    def test_data_types(self, data_loader):
        """Test that all feature columns are numeric"""
        df = data_loader.load_raw_data()
        feature_columns = ['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)']
        
        for col in feature_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"
    
    def test_target_distribution(self, data_loader):
        """Test that target classes are balanced"""
        df = data_loader.load_raw_data()
        target_counts = df['target'].value_counts()
        
        # Each class should have exactly 50 samples in IRIS dataset
        for count in target_counts.values:
            assert count == 50, "Target classes are not balanced"
