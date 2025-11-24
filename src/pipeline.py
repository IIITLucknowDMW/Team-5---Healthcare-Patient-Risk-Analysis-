"""
Main Pipeline for Healthcare Patient Risk Analysis

This module integrates outlier detection, clustering, and classification
to provide comprehensive patient risk analysis.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

from data_loader import HeartDiseaseDataLoader
from outlier_detection import OutlierDetector
from clustering import PatientClustering
from classification import DiseaseRiskClassifier


class PatientRiskAnalysisPipeline:
    """
    Complete pipeline for patient risk analysis.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the analysis pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.data_loader = HeartDiseaseDataLoader()
        self.outlier_detector = None
        self.clustering = None
        self.classifier = None
        
        self.original_data = None
        self.cleaned_data = None
        self.clustered_data = None
        
        self.results = {
            'timestamp': None,
            'outlier_detection': {},
            'clustering': {},
            'classification': {}
        }
    
    def load_data(self, filepath=None):
        """
        Load the dataset.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to data file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        self.original_data = self.data_loader.load_data(filepath)
        self.original_data = self.data_loader.preprocess()
        
        print(f"Data loaded successfully: {self.original_data.shape}")
        print(f"Columns: {self.original_data.columns.tolist()}")
        
        return self.original_data
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect and remove outliers.
        
        Parameters:
        -----------
        method : str
            Outlier detection method ('iqr' or 'zscore')
        threshold : float
            Threshold value
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned data
        dict
            Outlier statistics
        """
        print(f"\n{'='*60}")
        print("OUTLIER DETECTION")
        print(f"{'='*60}")
        
        self.outlier_detector = OutlierDetector(method=method, threshold=threshold)
        
        # Detect outliers on all numeric columns except target
        feature_columns = [col for col in self.original_data.columns 
                          if col != 'target' and self.original_data[col].dtype in [np.int64, np.float64]]
        
        self.cleaned_data, outlier_stats = self.outlier_detector.fit_transform(
            self.original_data, columns=feature_columns
        )
        
        print(f"\nOutlier Detection Method: {method}")
        print(f"Threshold: {threshold}")
        print(f"Original data size: {outlier_stats['original_size']}")
        print(f"Cleaned data size: {outlier_stats['cleaned_size']}")
        print(f"Total outliers removed: {outlier_stats['total_outliers']}")
        print(f"Percentage of outliers: {outlier_stats['outlier_percentage']:.2f}%")
        
        self.results['outlier_detection'] = outlier_stats
        
        return self.cleaned_data, outlier_stats
    
    def perform_clustering(self, n_clusters=3):
        """
        Perform K-Means clustering on patients.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        pandas.DataFrame
            Cluster statistics
        """
        print(f"\n{'='*60}")
        print("PATIENT CLUSTERING (K-MEANS)")
        print(f"{'='*60}")
        
        self.clustering = PatientClustering(n_clusters=n_clusters, random_state=self.random_state)
        
        # Use cleaned data or original if outlier detection wasn't performed
        data_to_cluster = self.cleaned_data if self.cleaned_data is not None else self.original_data
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in data_to_cluster.columns if col != 'target']
        
        # Fit clustering
        self.clustering.fit(data_to_cluster, feature_columns=feature_columns)
        
        # Get cluster statistics
        cluster_stats = self.clustering.get_cluster_statistics(data_to_cluster, feature_columns=feature_columns)
        
        # Evaluate clustering
        metrics = self.clustering.evaluate(data_to_cluster, feature_columns=feature_columns)
        
        print(f"\nNumber of clusters: {n_clusters}")
        print(f"\nCluster Statistics:")
        print(cluster_stats)
        print(f"\nClustering Metrics:")
        print(f"Inertia: {metrics['inertia']:.2f}")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        
        # Store results
        self.results['clustering'] = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_stats['cluster_size'].to_dict(),
            'metrics': metrics,
            'cluster_centers': self.clustering.cluster_centers.tolist()
        }
        
        # Add cluster labels to data
        self.clustered_data = data_to_cluster.copy()
        self.clustered_data['cluster'] = self.clustering.cluster_labels
        
        return cluster_stats
    
    def train_classifier(self, algorithm='random_forest', test_size=0.2):
        """
        Train classification model to predict disease risk.
        
        Parameters:
        -----------
        algorithm : str
            Classification algorithm to use
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        dict
            Classification results
        """
        print(f"\n{'='*60}")
        print("DISEASE RISK CLASSIFICATION")
        print(f"{'='*60}")
        
        self.classifier = DiseaseRiskClassifier(
            algorithm=algorithm,
            random_state=self.random_state
        )
        
        # Use cleaned data or original if outlier detection wasn't performed
        data_to_classify = self.cleaned_data if self.cleaned_data is not None else self.original_data
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.classifier.prepare_data(
            data_to_classify,
            target_column='target',
            test_size=test_size
        )
        
        # Train model
        print(f"\nTraining {algorithm} classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.classifier.evaluate(X_test, y_test)
        
        print(f"\nClassification Results:")
        print(f"Algorithm: {algorithm}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        # Get feature importance for tree-based models
        if algorithm in ['random_forest', 'gradient_boosting']:
            feature_importance = self.classifier.get_feature_importance()
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            self.results['classification']['feature_importance'] = feature_importance.to_dict('records')
        
        # Store results
        self.results['classification'] = {
            'algorithm': algorithm,
            'test_size': test_size,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        
        if 'roc_auc' in metrics:
            self.results['classification']['roc_auc'] = metrics['roc_auc']
        
        return metrics
    
    def run_complete_analysis(self, filepath=None, n_clusters=3, algorithm='random_forest'):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to data file
        n_clusters : int
            Number of clusters for K-Means
        algorithm : str
            Classification algorithm
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        self.results['timestamp'] = datetime.now().isoformat()
        
        print(f"\n{'#'*60}")
        print("PATIENT RISK ANALYSIS PIPELINE")
        print(f"{'#'*60}\n")
        
        # Step 1: Load data
        self.load_data(filepath)
        
        # Step 2: Detect and remove outliers
        self.detect_outliers(method='iqr', threshold=1.5)
        
        # Step 3: Perform clustering
        self.perform_clustering(n_clusters=n_clusters)
        
        # Step 4: Train classifier
        self.train_classifier(algorithm=algorithm)
        
        print(f"\n{'#'*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'#'*60}\n")
        
        return self.results
    
    def save_results(self, filepath='results/analysis_results.json'):
        """
        Save analysis results to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to {filepath}")
    
    def get_patient_profile(self, patient_data):
        """
        Get complete risk profile for a patient.
        
        Parameters:
        -----------
        patient_data : pandas.DataFrame or dict
            Patient data
            
        Returns:
        --------
        dict
            Patient risk profile
        """
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Get cluster assignment
        cluster = self.clustering.predict(patient_data)[0]
        
        # Get disease risk prediction
        risk_prediction = self.classifier.predict(patient_data)[0]
        risk_probability = self.classifier.predict_proba(patient_data)[0]
        
        profile = {
            'cluster': int(cluster),
            'risk_prediction': int(risk_prediction),
            'risk_probability': {
                'no_disease': float(risk_probability[0]),
                'disease': float(risk_probability[1])
            }
        }
        
        return profile


if __name__ == '__main__':
    # Run the complete analysis pipeline
    pipeline = PatientRiskAnalysisPipeline(random_state=42)
    results = pipeline.run_complete_analysis(n_clusters=3, algorithm='random_forest')
    
    # Save results
    pipeline.save_results('results/analysis_results.json')
