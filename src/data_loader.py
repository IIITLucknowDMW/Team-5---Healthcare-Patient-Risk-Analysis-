"""
Data Loader Module for Healthcare Patient Risk Analysis

This module handles loading and preprocessing the UCI Heart Disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml


class HeartDiseaseDataLoader:
    """
    Load and preprocess the UCI Heart Disease dataset.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = None
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        self.column_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (1-3)',
            'target': 'Heart disease presence (0 = no, 1-4 = yes)'
        }
    
    def load_data(self, filepath=None):
        """
        Load the heart disease dataset.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to CSV file. If None, creates sample data.
            
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset
        """
        if filepath:
            self.data = pd.read_csv(filepath)
        else:
            # Create sample data if no file is provided
            self.data = self._create_sample_data()
        
        return self.data
    
    def _create_sample_data(self):
        """
        Create sample heart disease data for demonstration.
        
        Returns:
        --------
        pandas.DataFrame
            Sample dataset
        """
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(150, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(80, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(1, 4, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target based on risk factors
        risk_score = (
            (df['age'] > 60).astype(int) * 0.3 +
            (df['sex'] == 1).astype(int) * 0.2 +
            (df['cp'] > 2).astype(int) * 0.3 +
            (df['trestbps'] > 140).astype(int) * 0.2 +
            (df['chol'] > 240).astype(int) * 0.2 +
            (df['thalach'] < 120).astype(int) * 0.3 +
            (df['exang'] == 1).astype(int) * 0.3 +
            (df['oldpeak'] > 2).astype(int) * 0.2 +
            np.random.uniform(0, 0.3, n_samples)
        )
        
        df['target'] = (risk_score > 1.0).astype(int)
        
        # Add some outliers
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df.loc[outlier_indices, 'chol'] = np.random.randint(500, 600, len(outlier_indices))
        df.loc[outlier_indices[:5], 'trestbps'] = np.random.randint(220, 280, 5)
        
        return df
    
    def get_feature_columns(self):
        """
        Get list of feature columns (excluding target).
        
        Returns:
        --------
        list
            List of feature column names
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return [col for col in self.data.columns if col != 'target']
    
    def get_data_info(self):
        """
        Get information about the dataset.
        
        Returns:
        --------
        dict
            Dataset information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'target_distribution': self.data['target'].value_counts().to_dict() if 'target' in self.data.columns else None,
            'basic_statistics': self.data.describe().to_dict()
        }
        
        return info
    
    def preprocess(self):
        """
        Basic preprocessing of the dataset.
        
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataset
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Handle missing values if any
        self.data = self.data.fillna(self.data.median(numeric_only=True))
        
        # Convert target to binary (0 = no disease, 1 = disease present)
        if 'target' in self.data.columns:
            self.data['target'] = (self.data['target'] > 0).astype(int)
        
        return self.data
