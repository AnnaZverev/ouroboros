import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_embedding_distribution(X_emb, save_path=None):
    """Plot histogram distributions of first 9 embedding dimensions."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i < X_emb.shape[1]:
            ax.hist(X_emb.iloc[:, i], bins=50, edgecolor='black')
            ax.set_title(f'Embedding dim {i}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
        else:
            ax.set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()

def plot_target_distributions(y, save_path=None):
    """Plot histogram of each target nutrient."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    for i, col in enumerate(y.columns):
        ax = axes[i]
        ax.hist(y[col].dropna(), bins=40, edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
    for i in range(len(y.columns), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()

def spatial_scatter(df, x_col='lon', y_col='lat', color_col=None, save_path=None):
    """Plot spatial distribution colored by a column."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col:
        scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], s=20, alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Spatial distribution{" colored by "+color_col if color_col else ""}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()

def correlation_heatmap(X, y, method='pearson', save_path=None):
    """Compute correlation between each embedding feature and each target. Plot heatmap."""
    # Combine for correlation
    combined = pd.concat([X, y], axis=1)
    corr_matrix = combined.corr(method=method)
    # Extract block: features vs targets
    feature_target_corr = corr_matrix.loc[X.columns, y.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(feature_target_corr, cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': f'{method} correlation'})
    ax.set_title('Feature-Target Correlations')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
    return feature_target_corr

def mutual_information_matrix(X, y, n_bins=10):
    """Compute mutual information between each feature and each target (discretized)."""
    from sklearn.metrics import mutual_info_score
    mi = pd.DataFrame(index=X.columns, columns=y.columns, dtype=float)
    for f in X.columns:
        # Discretize feature
        f_binned = pd.qcut(X[f], q=n_bins, labels=False, duplicates='drop')
        for t in y.columns:
            t_series = y[t].dropna()
            if len(t_series) == 0:
                continue
            # Align indexes
            common_idx = f_binned.index.intersection(t_series.index)
            if len(common_idx) == 0:
                continue
            t_binned = pd.qcut(t_series.loc[common_idx], q=n_bins, labels=False, duplicates='drop')
            mi.loc[f, t] = mutual_info_score(f_binned.loc[common_idx], t_binned)
    return mi

def pca_explained_variance_plot(X, save_path=None):
    """Plot cumulative explained variance vs number of components."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
    ax.axhline(0.95, ls='--', color='red', label='95% threshold')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('PCA Explained Variance (embeddings only)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
    return pca
