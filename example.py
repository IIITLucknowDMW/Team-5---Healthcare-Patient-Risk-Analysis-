#!/usr/bin/env python3
"""
Example script demonstrating the usage of the Patient Risk Analysis Pipeline.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import PatientRiskAnalysisPipeline
from visualization import create_comprehensive_report


def main():
    """Run a complete patient risk analysis example."""
    
    print("="*70)
    print(" HEALTHCARE PATIENT RISK ANALYSIS - DEMONSTRATION")
    print("="*70)
    
    # Initialize the pipeline
    pipeline = PatientRiskAnalysisPipeline(random_state=42)
    
    # Run complete analysis
    # You can provide a filepath to your own UCI Heart Disease dataset CSV
    # If no filepath is provided, sample data will be generated
    results = pipeline.run_complete_analysis(
        filepath=None,  # Set to your CSV path if available
        n_clusters=3,
        algorithm='random_forest'
    )
    
    # Save results
    pipeline.save_results('results/analysis_results.json')
    
    # Generate visualizations
    create_comprehensive_report(pipeline, save_dir='results')
    
    print("\n" + "="*70)
    print(" ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n1. OUTLIER DETECTION:")
    print(f"   - Original samples: {results['outlier_detection']['original_size']}")
    print(f"   - Outliers removed: {results['outlier_detection']['total_outliers']}")
    print(f"   - Clean samples: {results['outlier_detection']['cleaned_size']}")
    print(f"   - Outlier rate: {results['outlier_detection']['outlier_percentage']:.2f}%")
    
    print("\n2. CLUSTERING:")
    print(f"   - Number of clusters: {results['clustering']['n_clusters']}")
    print(f"   - Silhouette score: {results['clustering']['metrics']['silhouette_score']:.4f}")
    print(f"   - Cluster sizes: {results['clustering']['cluster_sizes']}")
    
    print("\n3. CLASSIFICATION:")
    print(f"   - Algorithm: {results['classification']['algorithm']}")
    print(f"   - Accuracy: {results['classification']['accuracy']:.4f}")
    print(f"   - Precision: {results['classification']['precision']:.4f}")
    print(f"   - Recall: {results['classification']['recall']:.4f}")
    print(f"   - F1-Score: {results['classification']['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nResults and visualizations have been saved to the 'results/' directory.")
    print("Check the following files:")
    print("  - results/analysis_results.json")
    print("  - results/clustering_results.png")
    print("  - results/cluster_profiles.png")
    print("  - results/classification_results.png")
    print("  - results/feature_importance.png")
    
    # Example: Get risk profile for a new patient
    print("\n" + "="*70)
    print(" EXAMPLE: PATIENT RISK PROFILE")
    print("="*70)
    
    sample_patient = {
        'age': 65,
        'sex': 1,
        'cp': 3,
        'trestbps': 150,
        'chol': 280,
        'fbs': 1,
        'restecg': 1,
        'thalach': 110,
        'exang': 1,
        'oldpeak': 3.5,
        'slope': 2,
        'ca': 2,
        'thal': 3
    }
    
    print("\nSample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    profile = pipeline.get_patient_profile(sample_patient)
    
    print("\nRisk Analysis:")
    print(f"  Cluster Assignment: {profile['cluster']}")
    print(f"  Risk Prediction: {'High Risk' if profile['risk_prediction'] == 1 else 'Low Risk'}")
    print(f"  Disease Probability: {profile['risk_probability']['disease']:.2%}")
    print(f"  No Disease Probability: {profile['risk_probability']['no_disease']:.2%}")


if __name__ == '__main__':
    main()
