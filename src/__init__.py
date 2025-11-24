"""
Healthcare Patient Risk Analysis Package

This package provides tools for automated patient risk stratification using:
- Outlier Detection: Clean medical data
- Clustering (K-Means): Identify patient symptom profiles
- Classification: Predict specific disease risks
"""

from .data_loader import HeartDiseaseDataLoader
from .outlier_detection import OutlierDetector
from .clustering import PatientClustering
from .classification import DiseaseRiskClassifier
from .pipeline import PatientRiskAnalysisPipeline

__all__ = [
    'HeartDiseaseDataLoader',
    'OutlierDetector',
    'PatientClustering',
    'DiseaseRiskClassifier',
    'PatientRiskAnalysisPipeline'
]

__version__ = '1.0.0'
