"""
Classification Module for Healthcare Patient Risk Analysis

This module provides multiple classification algorithms to predict specific disease risks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


class DiseaseRiskClassifier:
    """
    Classify patients into disease risk categories using multiple algorithms.
    """
    
    def __init__(self, algorithm='random_forest', random_state=42):
        """
        Initialize the DiseaseRiskClassifier.
        
        Parameters:
        -----------
        algorithm : str
            Classification algorithm to use
            ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm')
        random_state : int
            Random state for reproducibility
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model based on the algorithm."""
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5
            )
        elif self.algorithm == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1
            )
        elif self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.algorithm == 'svm':
            self.model = SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def prepare_data(self, data, target_column, feature_columns=None, test_size=0.2):
        """
        Prepare data for training and testing.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        target_column : str
            Name of the target column
        feature_columns : list, optional
            List of feature columns. If None, use all numeric columns except target.
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if feature_columns is None:
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                             if col != target_column]
        
        self.feature_columns = feature_columns
        
        X = data[feature_columns]
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """
        Train the classification model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features
        y_train : pandas.Series or numpy.ndarray
            Training labels
            
        Returns:
        --------
        self
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame or numpy.ndarray
            Test features
        y_test : pandas.Series or numpy.ndarray
            Test labels
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features
        y : pandas.Series or numpy.ndarray
            Labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': scores.tolist(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models).
        
        Returns:
        --------
        pandas.DataFrame
            Feature importance scores
        """
        if self.algorithm not in ['random_forest', 'gradient_boosting']:
            raise ValueError("Feature importance only available for tree-based models")
        
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
