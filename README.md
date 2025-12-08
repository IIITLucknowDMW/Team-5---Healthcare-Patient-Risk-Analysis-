# Healthcare Patient Risk Analysis - Multi-Algorithm Pipeline

A comprehensive data mining project that compares multiple machine learning algorithms at each stage of a healthcare risk analysis pipeline using the UCI Heart Disease Dataset.

## 📊 Pipeline Overview

This project implements a **3-stage pipeline** comparing different algorithms at each step:

### Stage 1: Outlier Detection
| Algorithm | Description |
|-----------|-------------|
| Z-Score | Statistical deviation method (threshold = 3σ) |
| Local Outlier Factor (LOF) | Density-based detection |
| **Isolation Forest** ✓ | Tree-based isolation (Selected) |

### Stage 2: Clustering
| Algorithm | Description |
|-----------|-------------|
| Hierarchical (Agglomerative) | Linkage-based clustering |
| DBSCAN | Density-based spatial clustering |
| **K-Means** ✓ | Centroid-based clustering (Selected) |

### Stage 3: Classification
| Algorithm | Description |
|-----------|-------------|
| Logistic Regression | Linear classifier |
| Support Vector Machine (SVM) | Margin-based classifier |
| **Random Forest** ✓ | Ensemble tree classifier (Selected) |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python risk_analysis_pipeline.py
```

## 📁 Output Files

After running the pipeline, the following files are generated:

### Visualizations
- `stage1_outlier_comparison.png` - Comparison of outlier detection methods
- `stage2_dendrogram.png` - Hierarchical clustering dendrogram
- `stage2_elbow_silhouette.png` - K-Means optimization plots
- `stage2_clustering_comparison.png` - PCA projection of all clustering methods
- `stage3_confusion_matrices.png` - Classifier comparison
- `stage3_metrics_comparison.png` - Bar chart of all metrics
- `stage3_feature_importance.png` - Random Forest feature importance

### Data Files
- `metrics.txt` - Complete metrics report
- `cluster_centers.txt` - K-Means cluster profiles

## 📈 Dataset

**UCI Heart Disease Dataset (Cleveland)**
- 303 patients
- 7 features: age, sex, cp, trestbps, chol, thalach, target
- Binary classification: Heart Disease (1) vs No Disease (0)

## 👨‍💻 Author

Data Mining Project - 7th Semester B.Tech
