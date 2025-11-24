"""
Visualization Module for Healthcare Patient Risk Analysis

This module provides visualization functions for the analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def plot_outlier_detection(data, outlier_detector, save_path=None):
    """
    Visualize outliers in the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original data
    outlier_detector : OutlierDetector
        Fitted outlier detector
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Outlier Detection Analysis', fontsize=16, fontweight='bold')
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2, idx % 2]
        
        # Box plot
        ax.boxplot(data[col].dropna(), vert=True)
        ax.set_title(f'{col} - Box Plot')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Outlier detection plot saved to {save_path}")
    
    plt.close()


def plot_clustering_results(data, clustering, save_path=None):
    """
    Visualize clustering results.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster labels
    clustering : PatientClustering
        Fitted clustering model
    save_path : str, optional
        Path to save the plot
    """
    # Prepare data
    feature_cols = [col for col in data.columns if col not in ['target', 'cluster']]
    X = data[feature_cols]
    
    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('K-Means Clustering Results', fontsize=16, fontweight='bold')
    
    # Scatter plot of clusters
    ax1 = axes[0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=clustering.cluster_labels, 
                         cmap='viridis', 
                         alpha=0.6, 
                         edgecolors='black',
                         s=50)
    ax1.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('Patient Clusters (PCA Projection)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Cluster size distribution
    ax2 = axes[1]
    cluster_counts = pd.Series(clustering.cluster_labels).value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Cluster Size Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(cluster_counts.values):
        ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustering plot saved to {save_path}")
    
    plt.close()


def plot_cluster_profiles(cluster_stats, save_path=None):
    """
    Plot cluster profiles showing mean feature values.
    
    Parameters:
    -----------
    cluster_stats : pandas.DataFrame
        Cluster statistics
    save_path : str, optional
        Path to save the plot
    """
    # Remove cluster_size column for plotting
    plot_data = cluster_stats.drop('cluster_size', axis=1, errors='ignore')
    
    # Normalize data for better visualization
    plot_data_norm = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(plot_data_norm.T, annot=False, cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Cluster Profiles (Normalized Feature Values)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster profiles plot saved to {save_path}")
    
    plt.close()


def plot_classification_results(metrics, save_path=None):
    """
    Visualize classification results.
    
    Parameters:
    -----------
    metrics : dict
        Classification metrics
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Disease Risk Classification Results', fontsize=16, fontweight='bold')
    
    # Confusion matrix
    ax1 = axes[0]
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Performance metrics
    ax2 = axes[1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    bars = ax2.barh(metric_names, metric_values, color='steelblue', edgecolor='black')
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Score')
    ax2.set_title('Classification Performance Metrics')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax2.text(value + 0.02, i, f'{value:.4f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification plot saved to {save_path}")
    
    plt.close()


def plot_feature_importance(feature_importance_df, top_n=10, save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features for Disease Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        ax.text(value + 0.005, i, f'{value:.4f}', 
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.close()


def create_comprehensive_report(pipeline, save_dir='results'):
    """
    Create comprehensive visualizations for the entire analysis.
    
    Parameters:
    -----------
    pipeline : PatientRiskAnalysisPipeline
        Fitted pipeline
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating comprehensive visualizations...")
    
    # 1. Clustering results
    if pipeline.clustering is not None and pipeline.clustered_data is not None:
        plot_clustering_results(
            pipeline.clustered_data,
            pipeline.clustering,
            save_path=f'{save_dir}/clustering_results.png'
        )
        
        cluster_stats = pipeline.clustering.get_cluster_statistics(
            pipeline.clustered_data.drop('cluster', axis=1),
            feature_columns=[col for col in pipeline.clustered_data.columns 
                           if col not in ['target', 'cluster']]
        )
        plot_cluster_profiles(
            cluster_stats,
            save_path=f'{save_dir}/cluster_profiles.png'
        )
    
    # 2. Classification results
    if pipeline.classifier is not None:
        # Reconstruct test data for metrics
        data_to_classify = pipeline.cleaned_data if pipeline.cleaned_data is not None else pipeline.original_data
        X_train, X_test, y_train, y_test = pipeline.classifier.prepare_data(
            data_to_classify, target_column='target', test_size=0.2
        )
        metrics = pipeline.classifier.evaluate(X_test, y_test)
        
        plot_classification_results(
            metrics,
            save_path=f'{save_dir}/classification_results.png'
        )
        
        # Feature importance (if available)
        if hasattr(pipeline.classifier, 'get_feature_importance'):
            try:
                feature_importance = pipeline.classifier.get_feature_importance()
                plot_feature_importance(
                    feature_importance,
                    save_path=f'{save_dir}/feature_importance.png'
                )
            except ValueError:
                pass  # Not a tree-based model
    
    print(f"\nAll visualizations saved to {save_dir}/")
