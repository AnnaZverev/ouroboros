#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

data_path = Path('/content/drive/MyDrive/Ouroboros/LUCAS_SOIL_2018_AlphaEarth_Embeddings_Wheat_clean.csv')
df = pd.read_csv(data_path)

print("=== CLEANED DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
embedding_cols = [f'A{i:02d}' for i in range(64)] + ['A63']
target_cols = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'N', 'K', 'CaCO3']

print(f"Embeddings: {len(embedding_cols)}")
print(f"Targets: {len(target_cols)}")

# 1. CORRELATION ANALYSIS
print("\n=== 1. CORRELATION ANALYSIS (Pearson) ===")
correlation_results = {}
for target in target_cols:
    if target not in df.columns:
        continue
    # Compute correlations with embeddings
    target_corr = df[embedding_cols].corrwith(df[target])
    abs_corr = target_corr.abs()
    top5 = abs_corr.sort_values(ascending=False).head(5)
    correlation_results[target] = {
        'top_positive': target_corr[top5.index].sort_values(ascending=False),
        'top_negative': target_corr[abs_corr.sort_values().head(5).index].sort_values()
    }
    print(f"\n{target}:")
    print("  Top positive correlations:")
    for idx, val in correlation_results[target]['top_positive'].items():
        print(f"    {idx}: {val:.3f}")
    print("  Top negative correlations:")
    for idx, val in correlation_results[target]['top_negative'].items():
        print(f"    {idx}: {val:.3f}")

# 2. PRINCIPAL COMPONENT ANALYSIS
print("\n=== 2. PCA ON EMBEDDINGS ===")
X_emb = df[embedding_cols].fillna(df[embedding_cols].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_emb)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"Components for 95% variance: {len(pca.explained_variance_ratio_)}")
print(f"Explained variance by first 5 PCs: {pca.explained_variance_ratio_[:5].sum()*100:.1f}%")

# Correlation of first 3 PCs with targets
pc_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
for target in target_cols:
    if target not in df.columns:
        continue
    corrs = pc_df.corrwith(df[target])
    print(f"  {target}: PC1={corrs['PC1']:.3f}, PC2={corrs['PC2']:.3f}, PC3={corrs['PC3']:.3f}")

# 3. MUTUAL INFORMATION
print("\n=== 3. MUTUAL INFORMATION (embeddings vs targets) ===")
mi_results = {}
for target in target_cols:
    if target not in df.columns or df[target].isnull().any():
        continue
    y = df[target].values
    # Use median imputation for embeddings
    X_filled = X_emb.fillna(X_emb.median()).values
    mi = mutual_info_regression(X_filled, y, random_state=0, n_neighbors=3)
    mi_series = pd.Series(mi, index=embedding_cols)
    top5_mi = mi_series.sort_values(ascending=False).head(5)
    mi_results[target] = top5_mi
    print(f"\n{target}:")
    for idx, val in top5_mi.items():
        print(f"  {idx}: {val:.3f}")

# 4. SPATIAL PATTERNS (if lat/lon exist)
if 'lat' in df.columns and 'lon' in df.columns:
    print("\n=== 4. SPATIAL DISTRIBUTION ===")
    print(f"Lat range: {df['lat'].min():.3f} to {df['lat'].max():.3f}")
    print(f"Lon range: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
    # Check spatial autocorrelation potential
    print("Spatial coverage suggests potential for geospatial features.")

# 5. EXTERNAL DATA OPPORTUNITIES
print("\n=== 5. FEATURE ENGINEERING OPPORTUNITIES ===")
print("From EDA, we can consider:")
print("  - Adding climate layers: WorldClim temperature/precipitation based on lat/lon")
print("  - Elevation: SRTM or DEM data")
print("  - Soil texture classes: Derived from Atlas of Soils Europe")
print("  - Land cover: ESA CCI or MODIS")
print("  - Distance to water bodies, urban areas")
print("Justification: embeddings capture spectral patterns; external data adds environmental context.")

print("\nCompleted EDA analysis. Ready for modeling approach design.")
