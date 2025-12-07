import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_path = "data/diabetes.csv"
output_path = "data/diabetes_processed.csv"

df = pd.read_csv(input_path)

df = df.drop_duplicates()
df = df.fillna(df.mean())

df = pd.read_csv(input_path)

target_col = "Outcome"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X = X.copy()
for col in zero_as_missing:
    X.loc[X[col] == 0, col] = np.nan

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

corr_matrix = X_imputed.corr(method="pearson")

print("\n=== Correlation matrix (features only) ===")
print(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, fmt=".2f",
            cbar_kws={"label": "Pearson r"})
plt.title("Feature Correlation Heatmap (after median imputation)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

pca = PCA(n_components=None, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_
explained_ratio = pca.explained_variance_ratio_

cum_explained_ratio = np.cumsum(explained_ratio)

print("\n=== PCA explained variance ratio by component ===")
for i, r in enumerate(explained_ratio, start=1):
    print(f"PC{i}: {r:.4f} (cumulative: {cum_explained_ratio[i-1]:.4f})")

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, index=X.columns,
                           columns=[f"PC{i}" for i in range(1, loadings.shape[1]+1)])

print("\n=== PCA loadings (feature contributions) ===")
print(loadings_df.round(3))

loadings_df.round(4).to_csv("pca_loadings.csv")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_ratio)+1), explained_ratio, "o-", label="Explained variance ratio")
plt.plot(range(1, len(cum_explained_ratio)+1), cum_explained_ratio, "o--", label="Cumulative")
plt.xticks(range(1, len(explained_ratio)+1))
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.title("PCA Scree Plot")
plt.legend()
plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=150)
plt.close()

if X_pca.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    palette = {0: "#1f77b4", 1: "#d62728"}
    for cls in [0, 1]:
        idx = (y == cls).values
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=25, alpha=0.7, label=f"Outcome={cls}", c=palette[cls])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (PC1 vs PC2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_2d_projection.png", dpi=150)
    plt.close()

print("\nArtifacts saved:")
print(" - correlation_heatmap.png")
print(" - pca_scree_plot.png")
print(" - pca_loadings.csv")

processed_df = pd.DataFrame(X_imputed, columns=X.columns)
processed_df["Outcome"] = y.values
processed_df.to_csv(output_path, index=False)
print(f" - {output_path}")
