"""
Outlier Detection Module for Healthcare Patient Risk Analysis

This module provides functionality to detect and handle outliers in medical data
using statistical methods (IQR and Z-score).
"""

import numpy as np
import pandas as pd
from scipy import stats


class OutlierDetector:
    """
    Detect and remove outliers from medical data using multiple methods.
    """
    
    def __init__(self, method='iqr', threshold=1.5):
        """
        Initialize the OutlierDetector.
        
        Parameters:
        -----------
        method : str
            Method to use for outlier detection ('iqr' or 'zscore')
        threshold : float
            Threshold value (1.5 for IQR, 3.0 for Z-score typically)
        """
        self.method = method
        self.threshold = threshold
        self.outlier_indices = None
        
    def detect_outliers_iqr(self, data, column):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        column : str
            Column name to check for outliers
            
        Returns:
        --------
        pandas.Series
            Boolean series indicating outliers
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    def detect_outliers_zscore(self, data, column):
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        column : str
            Column name to check for outliers
            
        Returns:
        --------
        pandas.Series
            Boolean series indicating outliers
        """
        # Create a boolean series aligned with the original data index
        outliers = pd.Series(False, index=data.index)
        non_null_mask = data[column].notna()
        
        if non_null_mask.sum() > 0:
            z_scores = np.abs(stats.zscore(data.loc[non_null_mask, column]))
            outliers.loc[non_null_mask] = z_scores > self.threshold
        
        return outliers
    
    def fit_transform(self, data, columns=None):
        """
        Detect and remove outliers from specified columns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        columns : list, optional
            List of columns to check for outliers. If None, use all numeric columns.
            
        Returns:
        --------
        pandas.DataFrame
            Data with outliers removed
        dict
            Statistics about outliers detected
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series([False] * len(data), index=data.index)
        outlier_stats = {}
        
        for column in columns:
            if self.method == 'iqr':
                col_outliers = self.detect_outliers_iqr(data, column)
            elif self.method == 'zscore':
                col_outliers = self.detect_outliers_zscore(data, column)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            outlier_mask = outlier_mask | col_outliers
            outlier_stats[column] = col_outliers.sum()
        
        self.outlier_indices = outlier_mask
        cleaned_data = data[~outlier_mask].copy()
        
        total_outliers = outlier_mask.sum()
        outlier_stats['total_outliers'] = total_outliers
        outlier_stats['original_size'] = len(data)
        outlier_stats['cleaned_size'] = len(cleaned_data)
        outlier_stats['outlier_percentage'] = (total_outliers / len(data)) * 100
        
        return cleaned_data, outlier_stats
