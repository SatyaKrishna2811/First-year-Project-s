import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, r2_score, mean_squared_error, silhouette_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv(r"D:\Work_From_Anywhere_Salary_Data.csv")

# ---------------------------- Data Preprocessing and Report ----------------------------

# Basic Info
print("Initial Dataset Overview:\n")
print(df.info())

# Preview of the data
print("\nSample Rows:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

# Describe numeric features
print("\nStatistical Summary (Numeric Columns):")
print(df.describe())

# Check for duplicates
duplicate_rows = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")

# Drop duplicates if any
if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")

# Check data types
print("\nData Types:\n", df.dtypes)

# Value counts for categorical columns (top 5 most frequent)
print("\nTop Value Counts for Categorical Columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:\n", df[col].value_counts().head(5))

# Optional: Outlier detection summary
numeric_cols = df.select_dtypes(include='number').columns
print("\nOutlier Summary (based on IQR method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# Correlation Matrix Preview (before transformation)
print("\nCorrelation with Target (if available):")
if 'Salary (Annual)' in df.columns:
    corr = df.corr(numeric_only=True)['Salary (Annual)'].sort_values(ascending=False)
    print(corr)

print("\n--- End of Data Cleaning Report ---\n")

# Save original company names
df_company = df[['Company', 'Job Satisfaction Score (1-10)']].copy()

# Drop irrelevant columns
df.drop(columns=['Company', 'Job Title', 'Tech Stack', 'Currency', 'Perks'], errors='ignore', inplace=True)

# Fill missing numeric values with median
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Check salary column
if 'Salary (Annual)' not in df.columns:
    raise ValueError("Missing 'Salary (Annual)' column in dataset.")

# Create price category and separate salary for regression
df['price_category'] = pd.qcut(df['Salary (Annual)'], q=3, labels=['Low', 'Medium', 'High'])
salary_target = df['Salary (Annual)']
df.drop(columns='Salary (Annual)', inplace=True)

# Features and target for classification
X = df.drop('price_category', axis=1)
y = df['price_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification models
models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf')
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    diff = abs(train_acc - test_acc)

    if train_acc > test_acc and diff > 0.1:
        fit_status = "Overfitting"
    elif train_acc < 0.6 and test_acc < 0.6:
        fit_status = "Underfitting"
    else:
        fit_status = "Well-fitted"

    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'classification_report': classification_report(y_test, y_test_pred, target_names=['Low', 'Medium', 'High']),
        'fit_status': fit_status
    }

# Print results
for name, result in results.items():
    print(f"\n{name}")
    print(f"Train Accuracy: {result['train_accuracy']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"Fit Status: {result['fit_status']}")
    print(result['classification_report'])

# Accuracy comparison plot
model_names = list(results.keys())
train_accuracies = [results[m]['train_accuracy'] for m in model_names]
test_accuracies = [results[m]['test_accuracy'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy')
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
plt.xticks(x, model_names, rotation=15)
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance (Random Forest)
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Linear Regression for salary prediction
X_lr = X.copy()
y_lr = salary_target
X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
X_lr_train_scaled = scaler.fit_transform(X_lr_train)
X_lr_test_scaled = scaler.transform(X_lr_test)

lr = LinearRegression()
lr.fit(X_lr_train_scaled, y_lr_train)
y_lr_pred = lr.predict(X_lr_test_scaled)

print("\nLinear Regression for Salary Prediction")
print(f"R^2 Score: {r2_score(y_lr_test, y_lr_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_lr_test, y_lr_pred)):.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_lr_test, y=y_lr_pred, alpha=0.6)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.tight_layout()
plt.show()

# Correlation Heatmap
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Clustering
X_clust = StandardScaler().fit_transform(df.drop(columns='price_category'))
sse, sils = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_clust)
    sse.append(km.inertia_)
    sils.append(silhouette_score(X_clust, labels))

print("\nSilhouette Scores for different k values:")
for k, s in zip(k_range, sils):
    print(f"k = {k}: Silhouette Score = {s:.4f}")

# Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Best K from Silhouette
best_k = k_range[np.argmax(sils)]
print(f"\nBest K: {best_k}, Silhouette Score: {max(sils):.4f}")

# Final Clustering
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df['Cluster_KMeans'] = km_final.fit_predict(X_clust)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_clust)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_KMeans', palette='Set2')
plt.title(f'KMeans Clusters (K={best_k}) via PCA')
plt.tight_layout()
plt.show()

# Hierarchical Clustering
linked = linkage(X_clust, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()

agglo = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df['Cluster_Hierarchical'] = agglo.fit_predict(X_clust)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_Hierarchical', palette='Set1')
plt.title(f'Hierarchical Clustering (K={best_k}) via PCA')
plt.tight_layout()
plt.show()

# Cluster by company job satisfaction
company_avg = df_company.groupby('Company').mean(numeric_only=True).dropna()
company_scaled = StandardScaler().fit_transform(company_avg)
company_avg['Cluster'] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(company_scaled)

plt.figure()
sns.barplot(x=company_avg.index, y='Job Satisfaction Score (1-10)', hue='Cluster', data=company_avg)
plt.xticks(rotation=90)
plt.title('Company Clusters by Job Satisfaction')
plt.tight_layout()
plt.show()

# Confusion Matrices
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("\nBest Parameters from GridSearchCV:", grid.best_params_)
print("Best Cross-Validated Score:", grid.best_score_)