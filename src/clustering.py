"""
Clustering Module for Healthcare Patient Risk Analysis

This module provides K-Means clustering functionality to identify patient symptom profiles.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class PatientClustering:
    """
    Cluster patients based on their symptoms and characteristics using K-Means.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize the PatientClustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        max_clusters : int
            Maximum number of clusters to test
            
        Returns:
        --------
        dict
            Dictionary with inertia and silhouette scores for different k values
        """
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        
        return {
            'k_values': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def fit(self, data, feature_columns=None):
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        feature_columns : list, optional
            List of columns to use for clustering. If None, use all numeric columns.
            
        Returns:
        --------
        self
        """
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        return self
    
    def predict(self, data, feature_columns=None):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        feature_columns : list, optional
            List of columns to use for prediction
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_statistics(self, data, feature_columns=None):
        """
        Get statistics for each cluster.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data with cluster labels
        feature_columns : list, optional
            List of columns to compute statistics for
            
        Returns:
        --------
        pandas.DataFrame
            Statistics for each cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cluster_stats = data_with_clusters.groupby('cluster')[feature_columns].mean()
        cluster_sizes = data_with_clusters['cluster'].value_counts().sort_index()
        cluster_stats['cluster_size'] = cluster_sizes
        
        return cluster_stats
    
    def evaluate(self, data, feature_columns=None):
        """
        Evaluate clustering quality.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        feature_columns : list, optional
            List of columns used for clustering
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns]
        X_scaled = self.scaler.transform(X)
        
        metrics = {
            'inertia': self.kmeans.inertia_,
            'silhouette_score': silhouette_score(X_scaled, self.cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(X_scaled, self.cluster_labels)
        }
        
        return metrics
