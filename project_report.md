# Healthcare Patient Risk Analysis System
## A Multi-Algorithm Data Mining Approach for Heart Disease Prediction

---

# Acknowledgements

I would like to express my sincere gratitude to my faculty advisor for their guidance and support throughout this Data Mining project. Special thanks to the UCI Machine Learning Repository for providing the Heart Disease dataset that made this research possible. I am also grateful to my peers for their valuable feedback and discussions that helped refine the methodology.

---

# Abstract

Heart disease remains one of the leading causes of mortality worldwide, making early detection critical for patient outcomes. This project presents a comprehensive **Healthcare Patient Risk Analysis System** using a multi-stage data mining pipeline applied to the UCI Heart Disease Dataset (Cleveland subset). The methodology encompasses three core stages: **(1) Outlier Detection** comparing Z-Score, Local Outlier Factor (LOF), and Isolation Forest algorithms; **(2) Clustering** comparing Hierarchical, DBSCAN, and K-Means algorithms for patient risk stratification; and **(3) Classification** comparing Logistic Regression, Support Vector Machine (SVM), and Random Forest classifiers. 

The pipeline successfully identified **5.26%** of patients as outliers using Isolation Forest, stratified patients into **3 distinct risk clusters** using K-Means (with disease rates ranging from 24% to 78%), and achieved a classification **Recall of 75.86%** using SVM—the critical metric for minimizing missed diagnoses. Key findings reveal that **Maximum Heart Rate** and **Chest Pain Type** are stronger predictors of heart disease than traditionally emphasized Cholesterol levels. This work demonstrates the effectiveness of ensemble approaches and the importance of algorithm comparison in medical data mining.

**Keywords:** Heart Disease Prediction, Data Mining, Outlier Detection, Clustering, Classification, Machine Learning, Healthcare Analytics

---

# 1. Introduction

## 1.1 Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of death globally, claiming an estimated 17.9 million lives each year according to the World Health Organization. Early identification of at-risk patients is crucial for preventive intervention, yet traditional diagnostic methods often fail to detect subtle patterns in patient data. The challenge lies in developing automated, accurate, and interpretable systems that can:

1. **Clean noisy medical data** by identifying erroneous or anomalous patient records
2. **Stratify patients** into meaningful risk groups for targeted interventions
3. **Predict heart disease** with high sensitivity to minimize missed diagnoses

## 1.2 Motivation

The motivation for this project stems from three critical observations:

1. **Clinical Need:** False negatives in heart disease detection can be life-threatening. A patient incorrectly classified as healthy may not receive timely treatment.

2. **Data Quality Issues:** Medical datasets often contain data entry errors, equipment malfunctions, and rare anomalies that can compromise predictive models.

3. **Algorithm Selection:** No single algorithm universally outperforms others across all datasets and objectives. A systematic comparison is essential for informed model selection.

## 1.3 Project Objectives

The primary objectives of this project are:

| # | Objective | Stage |
|---|-----------|-------|
| 1 | Compare outlier detection algorithms to identify and remove noisy patient records | Preprocessing |
| 2 | Compare clustering algorithms to discover inherent patient risk groups | Unsupervised Learning |
| 3 | Compare classification algorithms optimizing for **Recall** to minimize missed diagnoses | Supervised Learning |
| 4 | Identify the most important predictive features for heart disease | Feature Analysis |

## 1.4 Contributions

This project makes the following contributions:

- **Comprehensive Pipeline:** A modular, reproducible 3-stage pipeline for healthcare risk analysis
- **Algorithm Comparison:** Systematic evaluation of 9 algorithms (3 per stage) with quantitative metrics
- **Clinical Insights:** Identification of key predictive features challenging conventional assumptions
- **Open Source Implementation:** Complete Python codebase available for academic and research use

---

# 2. Literature Review

## 2.1 Outlier Detection in Healthcare

Outlier detection is critical in medical data preprocessing. Chandola et al. (2009) provide a comprehensive survey of anomaly detection techniques. For healthcare applications:

- **Statistical Methods (Z-Score):** Simple but assume normal distribution; may miss multivariate outliers
- **Density-based Methods (LOF):** Breunig et al. (2000) introduced LOF, effective for varying-density datasets
- **Isolation-based Methods:** Liu et al. (2008) proposed Isolation Forest, leveraging the isolation property of anomalies

## 2.2 Patient Clustering and Risk Stratification

Clustering enables discovery of patient subgroups without prior labels:

- **K-Means:** MacQueen (1967) introduced this centroid-based method; widely used due to simplicity and the Elbow Method for selecting k
- **Hierarchical Clustering:** Produces interpretable dendrograms; Ward's linkage minimizes intra-cluster variance
- **DBSCAN:** Ester et al. (1996) introduced density-based clustering; handles arbitrary shapes but sensitive to ε parameter

## 2.3 Heart Disease Classification

Numerous studies have applied machine learning to heart disease prediction:

| Study | Dataset | Best Algorithm | Accuracy |
|-------|---------|----------------|----------|
| Amin et al. (2019) | Cleveland | Naïve Bayes | 86.4% |
| Mohan et al. (2019) | Cleveland | HRFLM | 88.7% |
| This Project | Cleveland | SVM | 77.05% |

The focus on **Recall** rather than Accuracy distinguishes this work, prioritizing patient safety over overall accuracy.

---

# 3. Methodology

## 3.1 Dataset Description

**Dataset:** UCI Heart Disease Dataset (Cleveland Subset)

| Attribute | Description | Type |
|-----------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Gender (1=Male, 0=Female) | Categorical |
| `cp` | Chest pain type (1-4) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `target` | Heart disease presence (1=Yes, 0=No) | Binary |

**Statistics:**
- **Total Samples:** 304 patients
- **Class Distribution:** 165 No Disease (54.3%) | 139 Disease (45.7%)
- **Missing Values:** Handled via mean imputation

## 3.2 Data Preprocessing

The preprocessing pipeline includes:

```
Raw Data → Missing Value Imputation → Categorical Encoding → Feature Scaling → Clean Dataset
```

1. **Subset Selection:** Filter to Cleveland dataset only
2. **Missing Values:** Mean imputation for numerical features, mode imputation for categorical
3. **Encoding:** 
   - Sex: Male=1, Female=0
   - Chest Pain: typical angina=1, atypical angina=2, non-anginal=3, asymptomatic=4
4. **Scaling:** StandardScaler for algorithms sensitive to feature magnitude

## 3.3 Stage A: Outlier Detection (Unsupervised)

### 3.3.1 Z-Score Method
- **Principle:** Flag points where $|z| > 3$ standard deviations from mean
- **Parameter:** Threshold = 3σ
- **Limitation:** Univariate; ignores multivariate context

### 3.3.2 Local Outlier Factor (LOF)
- **Principle:** Compare local density of a point to neighbors
- **Parameters:** n_neighbors=20, contamination=0.05
- **Limitation:** Sensitive to neighborhood size selection

### 3.3.3 Isolation Forest
- **Principle:** Anomalies are isolated with fewer random splits
- **Parameters:** contamination=0.05, random_state=42
- **Advantage:** Efficient for high-dimensional data

## 3.4 Stage B: Clustering (Unsupervised)

### 3.4.1 Hierarchical Clustering (Agglomerative)
- **Principle:** Bottom-up merging based on linkage criterion
- **Parameters:** n_clusters=3, linkage='ward'
- **Output:** Dendrogram visualization

### 3.4.2 DBSCAN
- **Principle:** Density-based spatial clustering
- **Parameters:** eps=1.5, min_samples=5
- **Output:** Variable clusters + noise points

### 3.4.3 K-Means
- **Principle:** Minimize within-cluster sum of squares (Inertia)
- **Optimal k Selection:** Elbow Method
- **Parameters:** n_clusters=3, n_init=10

**Evaluation Metric:** Silhouette Score
$$S = \frac{b - a}{\max(a, b)}$$
where $a$ = mean intra-cluster distance, $b$ = mean nearest-cluster distance

## 3.5 Stage C: Classification (Supervised)

### 3.5.1 Logistic Regression
- **Principle:** Linear decision boundary via sigmoid function
- **Parameters:** max_iter=1000, random_state=42

### 3.5.2 Support Vector Machine (SVM)
- **Principle:** Maximum margin hyperplane with kernel trick
- **Parameters:** kernel='rbf', random_state=42

### 3.5.3 Random Forest
- **Principle:** Ensemble of decision trees with bagging
- **Parameters:** n_estimators=100, random_state=42

**Primary Metric:** Recall (Sensitivity)
$$Recall = \frac{TP}{TP + FN}$$

**Rationale:** In healthcare, minimizing False Negatives (missed diagnoses) is critical for patient safety.

---

# 4. Results & Experiments

## 4.1 Outlier Detection Results

| Algorithm | Outliers Detected | Percentage | Key Parameter |
|-----------|-------------------|------------|---------------|
| Z-Score | 7 | 2.30% | threshold=3 |
| LOF | 16 | 5.26% | contamination=0.05 |
| **Isolation Forest** | 16 | 5.26% | contamination=0.05 |

### Observations:
- Z-Score detected fewer outliers due to its univariate nature
- LOF and Isolation Forest showed consistent results at 5% contamination
- Outliers were primarily characterized by extreme Cholesterol and Blood Pressure values

### Selected Algorithm: **Isolation Forest**
**Rationale:** Tree-based approach efficiently handles high-dimensional medical data without distributional assumptions.

## 4.2 Clustering Visualization

### Algorithm Comparison

| Algorithm | Clusters | Silhouette Score | Notes |
|-----------|----------|------------------|-------|
| Hierarchical | 3 | 0.1653 | Ward linkage |
| DBSCAN | 2 | 0.2214 | 24 noise points |
| **K-Means** | 3 | 0.1914 | Inertia=1242.91 |

### K-Means Cluster Profiles (k=3)

| Cluster | Avg Age | Avg BP | Avg Chol | Avg Max HR | Disease Rate |
|---------|---------|--------|----------|------------|--------------|
| **0** (Medium Risk) | 58.70 | 136.18 | 272.56 | 147.49 | 31% |
| **1** (High Risk) | 58.44 | 132.36 | 245.47 | 134.64 | **78%** |
| **2** (Low Risk) | 47.26 | 127.84 | 228.57 | 166.12 | **24%** |

### Key Insights:
- **High-Risk Cluster (1):** Older patients with lowest Max Heart Rate and highest disease rate
- **Low-Risk Cluster (2):** Younger patients with highest Max Heart Rate
- Clear inverse relationship between Max Heart Rate and disease probability

### Selected Algorithm: **K-Means**
**Rationale:** Elbow Method provided quantifiable basis for k=3; clinically interpretable clusters enable actionable risk stratification.

## 4.3 Classification Performance

### Algorithm Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Logistic Regression | 0.7213 | 0.7308 | 0.6552 | 0.6909 |
| **SVM (RBF)** | **0.7705** | **0.7586** | **0.7586** | **0.7586** |
| Random Forest | 0.7377 | 0.7241 | 0.7241 | 0.7241 |

### Confusion Matrix (SVM - Best Model)

|  | Predicted: No Disease | Predicted: Disease |
|--|----------------------|-------------------|
| **Actual: No Disease** | 25 (TN) | 7 (FP) |
| **Actual: Disease** | 7 (FN) | 22 (TP) |

- **True Positives (TP):** 22 — Correctly identified disease cases
- **False Negatives (FN):** 7 — Missed diagnoses (minimized)
- **True Negatives (TN):** 25 — Correctly identified healthy patients
- **False Positives (FP):** 7 — False alarms (acceptable trade-off)

### Feature Importance (Random Forest)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | `thalach` (Max Heart Rate) | 0.2389 | **Strongest predictor** |
| 2 | `cp` (Chest Pain Type) | 0.1951 | Symptom indicator |
| 3 | `age` | 0.1890 | Age-related risk |
| 4 | `chol` (Cholesterol) | 0.1704 | Less predictive than expected |
| 5 | `trestbps` (Blood Pressure) | 0.1399 | Moderate importance |
| 6 | `sex` | 0.0667 | Lowest importance |

### Selected Algorithm: **SVM (RBF Kernel)**
**Rationale:** Highest Recall (75.86%) minimizes False Negatives, fulfilling the ethical requirement of catching all possible sick patients.

---

# 5. Discussion

## 5.1 Algorithm Selection Insights

### Outlier Detection
The Isolation Forest proved most suitable for this medical dataset due to its ability to handle multiple features simultaneously without assuming a particular distribution. The tree-based isolation mechanism naturally identifies patients with unusual combinations of vital signs.

### Clustering
While DBSCAN achieved the highest Silhouette Score (0.2214), it produced only 2 clusters with 24 noise points—less useful for clinical stratification. K-Means with k=3 provided balanced, interpretable clusters that directly map to clinical risk levels (Low, Medium, High).

### Classification
The SVM's superior Recall performance highlights its effectiveness in the healthcare domain where missing a positive case can have fatal consequences. The RBF kernel's ability to model non-linear decision boundaries proved advantageous for this complex medical data.

## 5.2 Clinical Implications

1. **Max Heart Rate as Key Predictor:** The feature importance analysis challenges the conventional emphasis on Cholesterol. Reduced maximum heart rate during exercise correlates strongly with heart disease.

2. **Risk Stratification:** The three identified clusters can inform hospital protocols:
   - **Cluster 1 (High Risk, 78%):** Immediate cardiology referral
   - **Cluster 0 (Medium Risk, 31%):** Regular monitoring
   - **Cluster 2 (Low Risk, 24%):** Preventive counseling

3. **Diagnostic Support:** With 75.86% Recall, the system catches approximately 3 out of 4 disease cases, serving as a valuable screening tool to prioritize patients for further testing.

## 5.3 Limitations

1. **Dataset Size:** 304 samples is relatively small for deep learning approaches
2. **Feature Set:** Limited to 6 features; additional biomarkers could improve accuracy
3. **Single Dataset:** Results may not generalize to other populations
4. **Binary Classification:** Does not capture disease severity levels

---

# 6. Conclusion & Future Work

## 6.1 Conclusion

This project successfully developed a **Healthcare Patient Risk Analysis System** using a multi-stage data mining pipeline. Key achievements include:

| Objective | Achievement |
|-----------|-------------|
| Outlier Detection | Isolation Forest identified 5.26% anomalous records |
| Patient Clustering | K-Means stratified patients into 3 risk groups (24%-78% disease rate) |
| Classification | SVM achieved 75.86% Recall, minimizing missed diagnoses |
| Feature Analysis | Max Heart Rate identified as strongest predictor |

The systematic comparison of 9 algorithms across 3 stages demonstrates the importance of algorithm selection in medical data mining. The focus on **Recall** over Accuracy reflects the ethical priority of patient safety in healthcare applications.

## 6.2 Future Work

1. **Deep Learning:** Implement neural networks for potentially higher accuracy
2. **Ensemble Methods:** Combine predictions from multiple classifiers
3. **Feature Engineering:** Incorporate additional clinical biomarkers
4. **Real-time Deployment:** Develop a web application for clinical use
5. **Cross-validation:** Implement k-fold cross-validation for robust evaluation
6. **Explainability:** Add SHAP or LIME for model interpretability
7. **Multi-class Classification:** Predict disease severity levels (0-4)

---

# References

1. Amin, M. S., Chiam, Y. K., & Varathan, K. D. (2019). Identification of significant features and data mining techniques in predicting heart disease. *Telematics and Informatics*, 36, 82-93.

2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104.

3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.

4. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226-231.

5. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.

6. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281-297.

7. Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. *IEEE Access*, 7, 81542-81554.

8. UCI Machine Learning Repository. Heart Disease Dataset. https://archive.ics.uci.edu/ml/datasets/heart+disease

---

# Appendix

## A. Generated Visualizations

The pipeline generates the following visualization files:

| File | Description |
|------|-------------|
| `stage1_outlier_comparison.png` | Side-by-side comparison of outlier detection methods |
| `stage2_dendrogram.png` | Hierarchical clustering dendrogram |
| `stage2_elbow_silhouette.png` | K-Means optimization (Elbow + Silhouette) |
| `stage2_clustering_comparison.png` | PCA projection of all clustering methods |
| `stage3_confusion_matrices.png` | Confusion matrices for all classifiers |
| `stage3_metrics_comparison.png` | Bar chart comparing classification metrics |
| `stage3_feature_importance.png` | Random Forest feature importance |

## B. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python risk_analysis_pipeline.py
```

## C. Project Structure

```
Project/
├── heart_disease_uci.csv          # UCI Heart Disease Dataset
├── risk_analysis_pipeline.py      # Main pipeline script
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── insights_report.md             # This report
├── project_report.md              # Academic report
├── metrics.txt                    # Complete metrics output
├── cluster_centers.txt            # K-Means cluster profiles
└── *.png                          # Generated visualizations
```

---

*Report Generated: December 2024*  
*Data Mining Project - B.Tech 7th Semester*
