import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Filter for Cleveland dataset
    df = df[df['dataset'] == 'Cleveland']
    
    relevant_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalch', 'num']
    df = df[relevant_cols].copy()
    
    # Rename columns
    df.rename(columns={'thalch': 'thalach', 'num': 'target'}, inplace=True)
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Handle missing values - convert columns to numeric
    cols_to_numeric = ['age', 'trestbps', 'chol', 'thalach']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[cols_to_numeric] = imputer.fit_transform(df[cols_to_numeric])
    
    # Encode categorical variables
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    cp_mapping = {
        'typical angina': 1,
        'atypical angina': 2,
        'non-anginal': 3,
        'asymptomatic': 4
    }
    df['cp'] = df['cp'].map(cp_mapping)
    # Fill any missing cp/sex with mode
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
    df['cp'] = df['cp'].fillna(df['cp'].mode()[0])

    print(f"Data loaded and cleaned. Shape: {df.shape}")
    return df

def outlier_detection(df):
    print("\n--- Technique A: Outlier Detection ---")
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    X = df[features]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(X_scaled)
    df['outlier'] = outliers # -1 for outlier, 1 for inlier
    
    n_outliers = (outliers == -1).sum()
    print(f"Detected {n_outliers} outliers.")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='chol', y='thalach', hue='outlier', palette={1: 'blue', -1: 'red'})
    plt.title('Outlier Detection (Isolation Forest)\nRed = Outlier')
    plt.xlabel('Cholesterol')
    plt.ylabel('Max Heart Rate')
    plt.savefig('outliers.png')
    print("Saved outliers.png")
    
    return df, X_scaled

def clustering(df, X_scaled):
    print("\n--- Technique B: Clustering ---")
    # Elbow Method
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('elbow_method.png')
    print("Saved elbow_method.png")
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters
    
    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title(f'Patient Clusters (k={optimal_k}) - PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('clusters.png')
    print("Saved clusters.png")
    
    # Analyze clusters
    print("Cluster Centers:")
    centers = df.groupby('cluster')[['age', 'trestbps', 'chol', 'thalach', 'target']].mean()
    print(centers)
    centers.to_csv('cluster_centers.txt')
    
    return df

def classification(df):
    print("\n--- Technique C: Classification ---")
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Calculate and display metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix - Disease Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved confusion_matrix.png")
    
    # Save metrics to file
    with open('metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\nClassification Report:\n")
        f.write(report)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(features)[indices])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")

def main():
    filepath = 'heart_disease_uci.csv'
    df = load_and_preprocess_data(filepath)
    
    df, X_scaled = outlier_detection(df)
    
    df = clustering(df, X_scaled)
    
    classification(df)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
