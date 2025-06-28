import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os
import logging

class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_raw_data(self):
        """Load IRIS dataset"""
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
        
        # Ensure raw data directory exists
        os.makedirs(os.path.dirname(self.config['data']['raw_data_path']), exist_ok=True)
        
        # Save raw data
        df.to_csv(self.config['data']['raw_data_path'], index=False)
        self.logger.info(f"Raw data saved to {self.config['data']['raw_data_path']}")
        
        return df
    
    def validate_data(self, df):
        """Validate the dataset"""
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values")
        
        # Check data types
        expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)', 'target']
        
        for col in expected_columns[:-1]:  # Exclude target for numeric check
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} is not numeric")
        
        # Check target values
        expected_targets = [0, 1, 2]
        if not set(df['target'].unique()) == set(expected_targets):
            raise ValueError("Target values are not as expected")
        
        # Check data shape
        if df.shape[0] != 150 or df.shape[1] < 5:
            raise ValueError("Dataset shape is not as expected")
        
        self.logger.info("Data validation passed")
        return True
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Separate features and target
        X = df.drop(['target', 'target_name'], axis=1)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # Save processed data
        processed_dir = self.config['data']['processed_data_path']
        os.makedirs(processed_dir, exist_ok=True)
        
        X_train_scaled.to_csv(f"{processed_dir}/X_train.csv", index=False)
        X_test_scaled.to_csv(f"{processed_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)
        
        self.logger.info("Data preprocessing completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
