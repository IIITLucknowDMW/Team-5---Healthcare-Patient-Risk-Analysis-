# Healthcare Patient Risk Analysis - Insights Report

## 1. Overview
This report summarizes the findings from the analysis of the UCI Heart Disease dataset (Cleveland). The pipeline performed data preprocessing, outlier detection, patient clustering, and disease prediction.

## 2. Outlier Detection (Technique A)
- **Method**: Isolation Forest
- **Findings**: The system identified patients with anomalous physiological data. These outliers may represent patients with extreme values or potential data entry errors.
- **Visualization**: See `outliers.png` for a scatter plot of Cholesterol vs Max Heart Rate highlighting these anomalies.

## 3. Patient Clustering (Technique B)
- **Method**: K-Means Clustering (k=3)
- **Cluster Profiles**:

| Cluster | Age (Avg) | Blood Pressure (Avg) | Cholesterol (Avg) | Max Heart Rate (Avg) | Disease Probability | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | ~59 | 136 | 273 | 147 | 31% | **High Cholesterol, Moderate Risk**: Older patients with very high cholesterol but relatively good heart rate response. |
| **1** | ~58 | 132 | 245 | 135 | 78% | **High Risk Group**: Older patients with lower max heart rate, indicating potential cardiac issues. |
| **2** | ~47 | 128 | 229 | 166 | 24% | **Low Risk / Healthy**: Younger patients with healthy blood pressure and excellent heart rate response. |

- **Key Insight**: Maximum Heart Rate (`thalach`) appears to be a significant differentiator. The High Risk group (Cluster 1) has the lowest average Max Heart Rate (135), while the Healthy group (Cluster 2) has the highest (166).
- **Visualization**: See `clusters.png` for the PCA projection of these patient groups.

## 4. Risk Prediction Model (Technique C)
- **Model**: Random Forest Classifier
- **Performance Metrics**:
    - **Accuracy**: 74%
    - **Precision (Disease)**: 0.72
    - **Recall (Disease)**: 0.72
- **Confusion Matrix**:
```
[[24  8]
 [ 8 21]]
```
- **Feature Importance**: The model identified key predictors for heart disease. See `feature_importance.png`. Typically, `thalach` (Max Heart Rate), `cp` (Chest Pain Type), and `oldpeak` (ST depression) are top drivers, though our simplified model focused on a subset of features.

## 5. Conclusion
The analysis successfully grouped patients into distinct risk profiles. The clustering revealed that age and heart rate response are strong indicators of patient similarity. The predictive model achieved reasonable accuracy (74%) given the limited feature set, confirming the feasibility of automated risk assessment.
