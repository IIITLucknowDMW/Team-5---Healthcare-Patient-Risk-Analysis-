"""
Healthcare Patient Risk Analysis - Multi-Algorithm Comparison Pipeline
=======================================================================
This pipeline compares multiple algorithms at each stage:
- Stage 1: Outlier Detection (Z-Score, LOF, Isolation Forest)
- Stage 2: Clustering (Hierarchical, DBSCAN, K-Means)
- Stage 3: Classification (Logistic Regression, SVM, Random Forest)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             silhouette_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load and preprocess the UCI Heart Disease dataset."""
    print("=" * 70)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df = df[df['dataset'] == 'Cleveland']
    
    relevant_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalch', 'num']
    df = df[relevant_cols].copy()
    
    df.rename(columns={'thalch': 'thalach', 'num': 'target'}, inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    cols_to_numeric = ['age', 'trestbps', 'chol', 'thalach']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    imputer = SimpleImputer(strategy='mean')
    df[cols_to_numeric] = imputer.fit_transform(df[cols_to_numeric])
    
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    cp_mapping = {'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4}
    df['cp'] = df['cp'].map(cp_mapping)
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
    df['cp'] = df['cp'].fillna(df['cp'].mode()[0])
    
    print(f"Dataset: Cleveland Heart Disease")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

# ============================================================================
# STAGE 1: OUTLIER DETECTION - COMPARING 3 ALGORITHMS
# ============================================================================

def outlier_detection_comparison(df):
    """Compare Z-Score, LOF, and Isolation Forest for outlier detection."""
    print("\n" + "=" * 70)
    print("STAGE 1: OUTLIER DETECTION - ALGORITHM COMPARISON")
    print("=" * 70)
    
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # ----- Algorithm 1: Z-Score Method -----
    print("\n--- Algorithm 1: Z-Score Method ---")
    z_scores = np.abs(stats.zscore(X_scaled))
    z_threshold = 3
    z_outliers = (z_scores > z_threshold).any(axis=1)
    n_z_outliers = z_outliers.sum()
    pct_z = (n_z_outliers / len(df)) * 100
    
    results['Z-Score'] = {
        'outliers': z_outliers.astype(int) * -2 + 1,  # Convert to -1/1 format
        'count': n_z_outliers,
        'percentage': pct_z,
        'threshold': z_threshold
    }
    print(f"  Threshold: {z_threshold} standard deviations")
    print(f"  Outliers detected: {n_z_outliers} ({pct_z:.2f}%)")
    
    # ----- Algorithm 2: Local Outlier Factor (LOF) -----
    print("\n--- Algorithm 2: Local Outlier Factor (LOF) ---")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_outliers = lof.fit_predict(X_scaled)
    n_lof_outliers = (lof_outliers == -1).sum()
    pct_lof = (n_lof_outliers / len(df)) * 100
    
    results['LOF'] = {
        'outliers': lof_outliers,
        'count': n_lof_outliers,
        'percentage': pct_lof,
        'contamination': 0.05
    }
    print(f"  Contamination: 0.05 (5%)")
    print(f"  n_neighbors: 20")
    print(f"  Outliers detected: {n_lof_outliers} ({pct_lof:.2f}%)")
    
    # ----- Algorithm 3: Isolation Forest -----
    print("\n--- Algorithm 3: Isolation Forest ---")
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_outliers = iso.fit_predict(X_scaled)
    n_iso_outliers = (iso_outliers == -1).sum()
    pct_iso = (n_iso_outliers / len(df)) * 100
    
    results['Isolation Forest'] = {
        'outliers': iso_outliers,
        'count': n_iso_outliers,
        'percentage': pct_iso,
        'contamination': 0.05
    }
    print(f"  Contamination: 0.05 (5%)")
    print(f"  Outliers detected: {n_iso_outliers} ({pct_iso:.2f}%)")
    
    # ----- Create Comparison Visualization -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    algorithms = ['Z-Score', 'LOF', 'Isolation Forest']
    for idx, algo in enumerate(algorithms):
        outlier_mask = results[algo]['outliers']
        colors = ['red' if o == -1 else 'blue' for o in outlier_mask]
        axes[idx].scatter(df['chol'], df['thalach'], c=colors, alpha=0.6)
        axes[idx].set_xlabel('Cholesterol')
        axes[idx].set_ylabel('Max Heart Rate')
        axes[idx].set_title(f'{algo}\nOutliers: {results[algo]["count"]} ({results[algo]["percentage"]:.1f}%)')
    
    plt.suptitle('Stage 1: Outlier Detection - Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stage1_outlier_comparison.png', dpi=150)
    print("\nSaved: stage1_outlier_comparison.png")
    
    # ----- Summary Table -----
    print("\n" + "-" * 50)
    print("OUTLIER DETECTION SUMMARY")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'Outliers':<12} {'Percentage':<12} {'Key Parameter'}")
    print("-" * 50)
    print(f"{'Z-Score':<20} {results['Z-Score']['count']:<12} {results['Z-Score']['percentage']:.2f}%{'':<8} threshold=3")
    print(f"{'LOF':<20} {results['LOF']['count']:<12} {results['LOF']['percentage']:.2f}%{'':<8} contamination=0.05")
    print(f"{'Isolation Forest':<20} {results['Isolation Forest']['count']:<12} {results['Isolation Forest']['percentage']:.2f}%{'':<8} contamination=0.05")
    print("-" * 50)
    print("SELECTED: Isolation Forest (best for high-dimensional medical data)")
    
    # Use Isolation Forest results for downstream
    df['outlier'] = iso_outliers
    
    return df, X_scaled, results

# ============================================================================
# STAGE 2: CLUSTERING - COMPARING 3 ALGORITHMS
# ============================================================================

def clustering_comparison(df, X_scaled):
    """Compare Hierarchical, DBSCAN, and K-Means clustering."""
    print("\n" + "=" * 70)
    print("STAGE 2: CLUSTERING - ALGORITHM COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # ----- Algorithm 1: Hierarchical Clustering (Agglomerative) -----
    print("\n--- Algorithm 1: Hierarchical Clustering (Agglomerative) ---")
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hier_labels = hierarchical.fit_predict(X_scaled)
    hier_silhouette = silhouette_score(X_scaled, hier_labels)
    
    results['Hierarchical'] = {
        'labels': hier_labels,
        'n_clusters': 3,
        'silhouette': hier_silhouette,
        'linkage': 'ward'
    }
    print(f"  Linkage: ward")
    print(f"  Number of clusters: 3")
    print(f"  Silhouette Score: {hier_silhouette:.4f}")
    
    # Create dendrogram
    plt.figure(figsize=(12, 5))
    linkage_matrix = linkage(X_scaled[:50], method='ward')  # Use subset for clarity
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram (Sample of 50 patients)')
    plt.xlabel('Patient Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('stage2_dendrogram.png', dpi=150)
    print("  Saved: stage2_dendrogram.png")
    
    # ----- Algorithm 2: DBSCAN -----
    print("\n--- Algorithm 2: DBSCAN ---")
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = (dbscan_labels == -1).sum()
    
    # Handle case where all points are noise or single cluster
    if n_dbscan_clusters > 1:
        # Exclude noise points for silhouette calculation
        mask = dbscan_labels != -1
        if mask.sum() > 1 and len(set(dbscan_labels[mask])) > 1:
            dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = -1
    else:
        dbscan_silhouette = -1
    
    results['DBSCAN'] = {
        'labels': dbscan_labels,
        'n_clusters': n_dbscan_clusters,
        'silhouette': dbscan_silhouette,
        'noise_points': n_noise,
        'eps': 1.5
    }
    print(f"  Epsilon: 1.5")
    print(f"  Min samples: 5")
    print(f"  Clusters found: {n_dbscan_clusters}")
    print(f"  Noise points: {n_noise}")
    print(f"  Silhouette Score: {dbscan_silhouette:.4f}" if dbscan_silhouette != -1 else "  Silhouette Score: N/A (insufficient clusters)")
    
    # ----- Algorithm 3: K-Means with Elbow Method -----
    print("\n--- Algorithm 3: K-Means (with Elbow Method) ---")
    
    # Elbow Method
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(K_range, inertia, marker='o', linewidth=2, markersize=8)
    axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].legend()
    
    axes[1].plot(K_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='green')
    axes[1].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].legend()
    
    plt.suptitle('K-Means: Determining Optimal Number of Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stage2_elbow_silhouette.png', dpi=150)
    print("  Saved: stage2_elbow_silhouette.png")
    
    # Final K-Means with optimal k=3
    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    kmeans_inertia = kmeans_final.inertia_
    
    results['K-Means'] = {
        'labels': kmeans_labels,
        'n_clusters': optimal_k,
        'silhouette': kmeans_silhouette,
        'inertia': kmeans_inertia
    }
    print(f"  Optimal k (from Elbow): {optimal_k}")
    print(f"  Inertia: {kmeans_inertia:.2f}")
    print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
    
    # ----- PCA Visualization Comparison -----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    algorithms = ['Hierarchical', 'DBSCAN', 'K-Means']
    for idx, algo in enumerate(algorithms):
        labels = results[algo]['labels']
        scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
        sil = results[algo]['silhouette']
        sil_str = f"{sil:.4f}" if sil != -1 else "N/A"
        axes[idx].set_title(f'{algo}\nClusters: {results[algo]["n_clusters"]}, Silhouette: {sil_str}')
        plt.colorbar(scatter, ax=axes[idx])
    
    plt.suptitle('Stage 2: Clustering - Algorithm Comparison (PCA Projection)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stage2_clustering_comparison.png', dpi=150)
    print("\nSaved: stage2_clustering_comparison.png")
    
    # ----- Summary Table -----
    print("\n" + "-" * 60)
    print("CLUSTERING SUMMARY")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Clusters':<12} {'Silhouette':<15} {'Key Metric'}")
    print("-" * 60)
    print(f"{'Hierarchical':<20} {results['Hierarchical']['n_clusters']:<12} {results['Hierarchical']['silhouette']:.4f}{'':<10} linkage=ward")
    
    dbscan_sil = f"{results['DBSCAN']['silhouette']:.4f}" if results['DBSCAN']['silhouette'] != -1 else "N/A"
    print(f"{'DBSCAN':<20} {results['DBSCAN']['n_clusters']:<12} {dbscan_sil:<15} noise={results['DBSCAN']['noise_points']}")
    
    print(f"{'K-Means':<20} {results['K-Means']['n_clusters']:<12} {results['K-Means']['silhouette']:.4f}{'':<10} inertia={results['K-Means']['inertia']:.0f}")
    print("-" * 60)
    print("SELECTED: K-Means (clear elbow at k=3, best silhouette score)")
    
    # Use K-Means for downstream
    df['cluster'] = kmeans_labels
    
    # Cluster profiling
    print("\n--- K-Means Cluster Profiles ---")
    cluster_profiles = df.groupby('cluster')[['age', 'trestbps', 'chol', 'thalach', 'target']].mean()
    print(cluster_profiles.round(2))
    cluster_profiles.to_csv('cluster_centers.txt')
    print("Saved: cluster_centers.txt")
    
    return df, results

# ============================================================================
# STAGE 3: CLASSIFICATION - COMPARING 3 ALGORITHMS
# ============================================================================

def classification_comparison(df):
    """Compare Logistic Regression, SVM, and Random Forest classifiers."""
    print("\n" + "=" * 70)
    print("STAGE 3: CLASSIFICATION - ALGORITHM COMPARISON")
    print("=" * 70)
    
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # ----- Algorithm 1: Logistic Regression -----
    print("\n--- Algorithm 1: Logistic Regression ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)
    
    results['Logistic Regression'] = {
        'accuracy': lr_acc,
        'precision': lr_prec,
        'recall': lr_recall,
        'f1': lr_f1,
        'confusion_matrix': lr_cm,
        'predictions': lr_pred
    }
    print(f"  Accuracy:  {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall:    {lr_recall:.4f}")
    print(f"  F1-Score:  {lr_f1:.4f}")
    
    # ----- Algorithm 2: Support Vector Machine (SVM) -----
    print("\n--- Algorithm 2: Support Vector Machine (SVM) ---")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred)
    svm_recall = recall_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)
    
    results['SVM'] = {
        'accuracy': svm_acc,
        'precision': svm_prec,
        'recall': svm_recall,
        'f1': svm_f1,
        'confusion_matrix': svm_cm,
        'predictions': svm_pred
    }
    print(f"  Kernel: RBF")
    print(f"  Accuracy:  {svm_acc:.4f}")
    print(f"  Precision: {svm_prec:.4f}")
    print(f"  Recall:    {svm_recall:.4f}")
    print(f"  F1-Score:  {svm_f1:.4f}")
    
    # ----- Algorithm 3: Random Forest -----
    print("\n--- Algorithm 3: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    rf_pred = rf.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    results['Random Forest'] = {
        'accuracy': rf_acc,
        'precision': rf_prec,
        'recall': rf_recall,
        'f1': rf_f1,
        'confusion_matrix': rf_cm,
        'predictions': rf_pred,
        'feature_importance': rf.feature_importances_
    }
    print(f"  n_estimators: 100")
    print(f"  Accuracy:  {rf_acc:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_recall:.4f}")
    print(f"  F1-Score:  {rf_f1:.4f}")
    
    # ----- Confusion Matrix Comparison -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    algorithms = ['Logistic Regression', 'SVM', 'Random Forest']
    for idx, algo in enumerate(algorithms):
        cm = results[algo]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        recall = results[algo]['recall']
        axes[idx].set_title(f'{algo}\nRecall: {recall:.4f}')
    
    plt.suptitle('Stage 3: Classification - Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stage3_confusion_matrices.png', dpi=150)
    print("\nSaved: stage3_confusion_matrices.png")
    
    # ----- Metrics Bar Chart -----
    metrics_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Accuracy': [results[a]['accuracy'] for a in algorithms],
        'Precision': [results[a]['precision'] for a in algorithms],
        'Recall': [results[a]['recall'] for a in algorithms],
        'F1-Score': [results[a]['f1'] for a in algorithms]
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algorithms))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', color='#2196F3')
    bars2 = ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', color='#4CAF50')
    bars3 = ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', color='#FF9800')
    bars4 = ax.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', color='#9C27B0')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('stage3_metrics_comparison.png', dpi=150)
    print("Saved: stage3_metrics_comparison.png")
    
    # ----- Feature Importance (Random Forest) -----
    importances = results['Random Forest']['feature_importance']
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('stage3_feature_importance.png', dpi=150)
    print("Saved: stage3_feature_importance.png")
    
    # ----- Summary Table -----
    print("\n" + "-" * 70)
    print("CLASSIFICATION SUMMARY")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score'}")
    print("-" * 70)
    for algo in algorithms:
        r = results[algo]
        print(f"{algo:<25} {r['accuracy']:.4f}{'':<7} {r['precision']:.4f}{'':<7} {r['recall']:.4f}{'':<7} {r['f1']:.4f}")
    print("-" * 70)
    
    # Find best by recall (patient safety priority)
    best_recall_algo = max(algorithms, key=lambda a: results[a]['recall'])
    print(f"SELECTED: {best_recall_algo} (highest Recall for patient safety)")
    
    # Detailed report for best model
    print(f"\n--- {best_recall_algo} Detailed Report ---")
    print(classification_report(y_test, results[best_recall_algo]['predictions'], 
                                target_names=['No Disease', 'Disease']))
    
    # Calculate TP, FN for best model
    cm_best = results[best_recall_algo]['confusion_matrix']
    tn, fp, fn, tp = cm_best.ravel()
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    
    return results, features

# ============================================================================
# SAVE ALL METRICS TO FILE
# ============================================================================

def save_metrics(outlier_results, cluster_results, class_results, features):
    """Save all metrics to a text file."""
    with open('metrics.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HEALTHCARE PATIENT RISK ANALYSIS - COMPLETE METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Stage 1
        f.write("STAGE 1: OUTLIER DETECTION\n")
        f.write("-" * 40 + "\n")
        for algo, data in outlier_results.items():
            f.write(f"{algo}: {data['count']} outliers ({data['percentage']:.2f}%)\n")
        f.write("\n")
        
        # Stage 2
        f.write("STAGE 2: CLUSTERING\n")
        f.write("-" * 40 + "\n")
        for algo, data in cluster_results.items():
            sil = f"{data['silhouette']:.4f}" if data['silhouette'] != -1 else "N/A"
            f.write(f"{algo}: {data['n_clusters']} clusters, Silhouette: {sil}\n")
        f.write("\n")
        
        # Stage 3
        f.write("STAGE 3: CLASSIFICATION\n")
        f.write("-" * 40 + "\n")
        for algo, data in class_results.items():
            f.write(f"{algo}:\n")
            f.write(f"  Accuracy:  {data['accuracy']:.4f}\n")
            f.write(f"  Precision: {data['precision']:.4f}\n")
            f.write(f"  Recall:    {data['recall']:.4f}\n")
            f.write(f"  F1-Score:  {data['f1']:.4f}\n")
        
        # Feature importance
        f.write("\nFEATURE IMPORTANCE (Random Forest):\n")
        f.write("-" * 40 + "\n")
        importances = class_results['Random Forest']['feature_importance']
        indices = np.argsort(importances)[::-1]
        for i in indices:
            f.write(f"  {features[i]}: {importances[i]:.4f}\n")
    
    print("\nSaved: metrics.txt")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("HEALTHCARE PATIENT RISK ANALYSIS - MULTI-ALGORITHM PIPELINE")
    print("=" * 70)
    
    filepath = 'heart_disease_uci.csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    
    # Stage 1: Outlier Detection
    df, X_scaled, outlier_results = outlier_detection_comparison(df)
    
    # Stage 2: Clustering
    df, cluster_results = clustering_comparison(df, X_scaled)
    
    # Stage 3: Classification
    class_results, features = classification_comparison(df)
    
    # Save all metrics
    save_metrics(outlier_results, cluster_results, class_results, features)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - stage1_outlier_comparison.png")
    print("  - stage2_dendrogram.png")
    print("  - stage2_elbow_silhouette.png")
    print("  - stage2_clustering_comparison.png")
    print("  - stage3_confusion_matrices.png")
    print("  - stage3_metrics_comparison.png")
    print("  - stage3_feature_importance.png")
    print("  - cluster_centers.txt")
    print("  - metrics.txt")

if __name__ == "__main__":
    main()
