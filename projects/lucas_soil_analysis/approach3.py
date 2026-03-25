from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import numpy as np
import pandas as pd

def engineer_features(X, embed_cols, geo_cols):
    """
    Create hybrid features:
    - Embedding stats: mean, std, min, max across dimensions
    - Embedding L2 norm
    - Geospatial interaction: elevation * latitude, elevation * longitude
    - Geospatial clusters: apply PCA to geo only (2 components)
    Returns augmented DataFrame and list of new feature names.
    """
    X_aug = X.copy()
    # Embedding statistics
    X_aug['emb_mean'] = X[embed_cols].mean(axis=1)
    X_aug['emb_std'] = X[embed_cols].std(axis=1)
    X_aug['emb_min'] = X[embed_cols].min(axis=1)
    X_aug['emb_max'] = X[embed_cols].max(axis=1)
    X_aug['emb_l2'] = np.linalg.norm(X[embed_cols].values, axis=1)
    
    # Geospatial interactions
    if 'elevation' in geo_cols:
        if 'lat' in geo_cols:
            X_aug['elev_lat'] = X['elevation'] * X['lat']
        if 'lon' in geo_cols:
            X_aug['elev_lon'] = X['elevation'] * X['lon']
    
    # Geo PCA (2 components)
    geo_data = X[geo_cols].fillna(X[geo_cols].median())
    scaler_geo = StandardScaler()
    geo_scaled = scaler_geo.fit_transform(geo_data)
    pca_geo = PCA(n_components=2)
    geo_pca = pca_geo.fit_transform(geo_scaled)
    X_aug['geo_pca1'] = geo_pca[:, 0]
    X_aug['geo_pca2'] = geo_pca[:, 1]
    
    # Mark new features
    new_features = ['emb_mean', 'emb_std', 'emb_min', 'emb_max', 'emb_l2',
                   'elev_lat', 'elev_lon', 'geo_pca1', 'geo_pca2']
    new_features = [f for f in new_features if f in X_aug.columns]
    return X_aug, new_features

def train_hybrid_enhanced(X, y, random_state=42):
    """
    Approach 3: Manual feature engineering + MLP (simple neural network).
    Returns: model, scaler, feature_names, metrics
    """
    X_aug, new_features = engineer_features(X)
    feature_names = list(X_aug.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    
    # Use a two-layer MLP with sklearn
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=random_state,
        verbose=False
    )
    
    metrics = {}
    for target in y.columns:
        y_target = y[target].dropna()
        if y_target.empty:
            continue
        common_idx = y_target.index.intersection(X_aug.index)
        X_target = X_scaled[np.searchsorted(X_aug.index, common_idx)] if X_aug.index.is_monotonic_increasing else X_scaled[[list(X_aug.index).index(i) for i in common_idx]]
        y_target = y_target.loc[common_idx]
        
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = cross_val_score(model, X_target, y_target, cv=cv, scoring='r2', n_jobs=-1)
        model.fit(X_target, y_target)
        pred = model.predict(X_target)
        metrics[target] = {
            'r2_cv_mean': r2_scores.mean(),
            'r2_cv_std': r2_scores.std(),
            'rmse': np.sqrt(mean_squared_error(y_target, pred)),
            'mae': mean_absolute_error(y_target, pred),
            'n_features': X_target.shape[1],
            'approach': 'HybridEnhanced'
        }
    return model, scaler, feature_names, metrics

def predict_hybrid(model, scaler, X_new, feature_names):
    """Predict using hybrid model: apply same feature engineering then scale."""
    # Need to reconstruct engineered features exactly as in training
    # We'll store the feature_names and scaler; we may need to re-run engineer_features on X_new with the same embedding/geo column references
    # For simplicity, we assume X_new has the same raw columns plus we can compute engineering again
    # Since engineer_features uses embed_cols and geo_cols from original context, we'll need to pass them or recompute similarly.
    # We'll accept X_new as raw features and apply engineer_features inside this function using heuristics to find embedding and geo columns.
    embed_cols = [c for c in X_new.columns if c.startswith('A') and c[1:].isdigit()]
    geo_cols = [c for c in X_new.columns if c not in embed_cols and c not in ['emb_mean','emb_std','emb_min','emb_max','emb_l2','elev_lat','elev_lon','geo_pca1','geo_pca2']]
    X_aug, _ = engineer_features(X_new, embed_cols, geo_cols)
    # Ensure columns are in same order as during training
    X_aug = X_aug.reindex(columns=feature_names, fill_value=np.nan)
    X_scaled = scaler.transform(X_aug)
    return model.predict(X_scaled)
