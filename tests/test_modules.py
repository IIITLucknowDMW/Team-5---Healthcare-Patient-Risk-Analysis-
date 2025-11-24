"""
Unit tests for the Patient Risk Analysis modules.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import HeartDiseaseDataLoader
from outlier_detection import OutlierDetector
from clustering import PatientClustering
from classification import DiseaseRiskClassifier


class TestDataLoader(unittest.TestCase):
    """Test the HeartDiseaseDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HeartDiseaseDataLoader()
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        data = self.loader.load_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('target', data.columns)
    
    def test_preprocess(self):
        """Test data preprocessing."""
        self.loader.load_data()
        preprocessed = self.loader.preprocess()
        self.assertIsNotNone(preprocessed)
        # Check that target is binary
        unique_targets = preprocessed['target'].unique()
        self.assertTrue(all(t in [0, 1] for t in unique_targets))


class TestOutlierDetector(unittest.TestCase):
    """Test the OutlierDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100)
        })
        # Add some outliers
        self.data.loc[0, 'feature1'] = 500
        self.data.loc[1, 'feature2'] = 200
    
    def test_iqr_detection(self):
        """Test IQR outlier detection."""
        detector = OutlierDetector(method='iqr', threshold=1.5)
        cleaned_data, stats = detector.fit_transform(self.data)
        
        self.assertLess(len(cleaned_data), len(self.data))
        self.assertGreater(stats['total_outliers'], 0)
        self.assertEqual(stats['original_size'], len(self.data))
    
    def test_zscore_detection(self):
        """Test Z-score outlier detection."""
        detector = OutlierDetector(method='zscore', threshold=3.0)
        cleaned_data, stats = detector.fit_transform(self.data)
        
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertIn('total_outliers', stats)


class TestPatientClustering(unittest.TestCase):
    """Test the PatientClustering class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100),
            'feature3': np.random.normal(75, 8, 100)
        })
    
    def test_fit_clustering(self):
        """Test fitting clustering model."""
        clustering = PatientClustering(n_clusters=3, random_state=42)
        clustering.fit(self.data)
        
        self.assertIsNotNone(clustering.cluster_labels)
        self.assertEqual(len(clustering.cluster_labels), len(self.data))
        self.assertEqual(len(np.unique(clustering.cluster_labels)), 3)
    
    def test_predict(self):
        """Test predicting cluster labels."""
        clustering = PatientClustering(n_clusters=3, random_state=42)
        clustering.fit(self.data)
        
        predictions = clustering.predict(self.data[:10])
        self.assertEqual(len(predictions), 10)
    
    def test_evaluate(self):
        """Test clustering evaluation."""
        clustering = PatientClustering(n_clusters=3, random_state=42)
        clustering.fit(self.data)
        
        metrics = clustering.evaluate(self.data)
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_score', metrics)
        self.assertIn('inertia', metrics)


class TestDiseaseRiskClassifier(unittest.TestCase):
    """Test the DiseaseRiskClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(100, 10, 200),
            'feature2': np.random.normal(50, 5, 200),
            'feature3': np.random.normal(75, 8, 200),
            'target': np.random.randint(0, 2, 200)
        })
    
    def test_prepare_data(self):
        """Test data preparation."""
        classifier = DiseaseRiskClassifier(algorithm='random_forest', random_state=42)
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            self.data, 'target', test_size=0.2
        )
        
        self.assertEqual(len(X_train) + len(X_test), len(self.data))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        classifier = DiseaseRiskClassifier(algorithm='random_forest', random_state=42)
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            self.data, 'target', test_size=0.2
        )
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_evaluate(self):
        """Test model evaluation."""
        classifier = DiseaseRiskClassifier(algorithm='random_forest', random_state=42)
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            self.data, 'target', test_size=0.2
        )
        
        classifier.fit(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_multiple_algorithms(self):
        """Test different classification algorithms."""
        algorithms = ['random_forest', 'logistic_regression', 'gradient_boosting']
        
        for algorithm in algorithms:
            classifier = DiseaseRiskClassifier(algorithm=algorithm, random_state=42)
            X_train, X_test, y_train, y_test = classifier.prepare_data(
                self.data, 'target', test_size=0.2
            )
            
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            
            self.assertEqual(len(predictions), len(X_test))


if __name__ == '__main__':
    unittest.main()
