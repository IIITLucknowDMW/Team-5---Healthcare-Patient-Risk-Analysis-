# Healthcare Patient Risk Analysis

## Overview

This project automates **patient risk stratification** using the UCI Heart Disease dataset. It features three core components that work together to provide comprehensive healthcare analytics:

1. **Outlier Detection** - Cleans medical data using statistical methods (IQR and Z-score)
2. **Clustering (K-Means)** - Identifies patient symptom profiles and groups similar patients
3. **Classification** - Predicts specific disease risks using multiple machine learning algorithms

## Features

- **Automated Data Cleaning**: Detects and removes outliers from medical data
- **Patient Segmentation**: Groups patients with similar characteristics using K-Means clustering
- **Risk Prediction**: Classifies patients into risk categories using state-of-the-art ML algorithms
- **Comprehensive Visualization**: Generates detailed plots and charts for analysis
- **Flexible Pipeline**: Easy-to-use API for complete end-to-end analysis
- **Multiple Algorithms**: Supports Random Forest, Gradient Boosting, Logistic Regression, and SVM

## Project Structure

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── outlier_detection.py    # Outlier detection module
│   ├── clustering.py            # K-Means clustering module
│   ├── classification.py        # Classification algorithms
│   ├── pipeline.py              # Main analysis pipeline
│   └── visualization.py         # Visualization utilities
├── data/                        # Data directory
├── results/                     # Analysis results and plots
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── example.py                   # Example usage script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/IIITLucknowDMW/Team-5---Healthcare-Patient-Risk-Analysis-.git
cd Team-5---Healthcare-Patient-Risk-Analysis-
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the example script:

```bash
python example.py
```

This will:
- Load or generate sample heart disease data
- Detect and remove outliers
- Perform K-Means clustering
- Train a Random Forest classifier
- Generate visualizations
- Save results to the `results/` directory

### Using the Pipeline in Your Code:

```python
from src.pipeline import PatientRiskAnalysisPipeline

# Initialize pipeline
pipeline = PatientRiskAnalysisPipeline(random_state=42)

# Run complete analysis
results = pipeline.run_complete_analysis(
    filepath='path/to/heart_disease.csv',  # Optional
    n_clusters=3,
    algorithm='random_forest'
)

# Get risk profile for a new patient
patient_data = {
    'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 150,
    'chol': 280, 'fbs': 1, 'restecg': 1, 'thalach': 110,
    'exang': 1, 'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 3
}
profile = pipeline.get_patient_profile(patient_data)
print(f"Risk Prediction: {profile['risk_prediction']}")
print(f"Disease Probability: {profile['risk_probability']['disease']:.2%}")
```

## Components

### 1. Outlier Detection

The `OutlierDetector` class provides two methods for detecting outliers:

- **IQR (Interquartile Range)**: Robust method for outlier detection
- **Z-Score**: Statistical method based on standard deviations

```python
from src.outlier_detection import OutlierDetector

detector = OutlierDetector(method='iqr', threshold=1.5)
cleaned_data, stats = detector.fit_transform(data)
```

### 2. Patient Clustering

The `PatientClustering` class uses K-Means to identify patient groups:

```python
from src.clustering import PatientClustering

clustering = PatientClustering(n_clusters=3)
clustering.fit(data)
cluster_stats = clustering.get_cluster_statistics(data)
metrics = clustering.evaluate(data)
```

### 3. Disease Risk Classification

The `DiseaseRiskClassifier` supports multiple algorithms:

```python
from src.classification import DiseaseRiskClassifier

# Options: 'random_forest', 'gradient_boosting', 'logistic_regression', 'svm'
classifier = DiseaseRiskClassifier(algorithm='random_forest')

X_train, X_test, y_train, y_test = classifier.prepare_data(data, 'target')
classifier.fit(X_train, y_train)
metrics = classifier.evaluate(X_test, y_test)
```

## Dataset

The project uses the **UCI Heart Disease dataset**, which contains the following features:

- **age**: Age in years
- **sex**: Sex (1 = male; 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (1-3)
- **target**: Heart disease presence (0 = no, 1 = yes)

## Results

The pipeline generates the following outputs:

1. **analysis_results.json**: Complete analysis metrics and statistics
2. **clustering_results.png**: Visualization of patient clusters
3. **cluster_profiles.png**: Heatmap of cluster characteristics
4. **classification_results.png**: Confusion matrix and performance metrics
5. **feature_importance.png**: Most important features for prediction

## Evaluation Metrics

### Clustering Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Score**: Lower is better for cluster separation
- **Inertia**: Within-cluster sum of squares

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve (for binary classification)

## Requirements

- Python 3.8+
- pandas 2.1.4
- numpy 1.26.2
- scikit-learn 1.3.2
- matplotlib 3.8.2
- seaborn 0.13.0
- scipy 1.11.4

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational purposes as part of the IIIT Lucknow Data Mining and Warehousing course.

## Authors

Team 5 - IIIT Lucknow

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- scikit-learn community for machine learning tools