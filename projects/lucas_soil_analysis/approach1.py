from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import numpy as np
import pandas as pd

def train_pca_gbdt(X, y, n_components=0.95, random_state=42):
    """
    Approach 1: PCA on embeddings + GBDT.
    - Standardize features
    - PCA on embedding subset (preserve variance)
    - Concatenate PCA components + geospatial features
    - Train LightGBM regression
    Returns: trained model, metrics dict, feature_names
    """
    X_emb = X.filter(like='A')
    X_geo = X.drop(columns=X_emb.columns)
    
    scaler = StandardScaler()
    X_emb_scaled = scaler.fit_transform(X_emb)
    
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_emb_scaled)
    pca_feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    
    # Combine
    X_combined = np.hstack([X_pca, X_geo.values])
    combined_feature_names = pca_feature_names + list(X_geo.columns)
    
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Cross-validation for each target
    metrics = {}
    for target in y.columns:
        y_target = y[target].dropna()
        if y_target.empty:
            continue
        # Align indexes
        common_idx = y_target.index.intersection(pd.DataFrame(X_combined, index=X.index).index)
        X_target = pd.DataFrame(X_combined, index=X.index).loc[common_idx]
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
            'approach': 'PCA+GBDT'
        }
    return model, pca, scaler, metrics, combined_feature_names

def predict_pca_gbdt(model, pca, scaler, X_new, X_geo_columns):
    """Predict using trained PCA+GBDT pipeline."""
    X_emb = X_new.filter(like='A')
    X_geo = X_new[X_geo_columns]
    X_emb_scaled = scaler.transform(X_emb)
    X_pca = pca.transform(X_emb_scaled)
    X_combined = np.hstack([X_pca, X_geo.values])
    return model.predict(X_combined)
