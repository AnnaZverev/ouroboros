from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import numpy as np
import pandas as pd

def train_mixture_of_experts(X, y, n_clusters=4, random_state=42):
    """
    Approach 2: Mixture of Experts.
    - Cluster the embedding space (KMeans or GMM)
    - Train separate LightGBM model for each cluster
    - Predict: find nearest cluster, then use its expert
    Returns: dict with clusterer, experts (one per target), scaler, metrics, cluster info
    """
    X_emb = X.filter(like='A')
    scaler = StandardScaler()
    X_emb_scaled = scaler.fit_transform(X_emb)
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = clusterer.fit_predict(X_emb_scaled)
    
    experts = {target: [] for target in y.columns}
    metrics = {}
    
    for target in y.columns:
        y_target = y[target].dropna()
        if y_target.empty:
            continue
        common_idx = y_target.index.intersection(X.index)
        X_common = X.loc[common_idx]
        clusters_common = clusters[np.searchsorted(X.index, common_idx)] if X.index.is_monotonic_increasing else clusters[[list(X.index).index(i) for i in common_idx]]
        # Train one model per cluster
        models_c = []
        for c in range(n_clusters):
            mask = clusters_common == c
            if mask.sum() < 10:  # too few samples, use global model fallback
                models_c.append(None)
                continue
            X_c = X_common.iloc[mask]
            y_c = y_target.loc[X_common.iloc[mask].index]
            model_c = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1
            )
            model_c.fit(X_c, y_c)
            models_c.append(model_c)
        experts[target] = models_c
        
        # Cross-validation for target: cluster assignment inside CV folds
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, test_idx in cv.split(X_common):
            X_tr = X_common.iloc[train_idx]
            X_te = X_common.iloc[test_idx]
            y_tr = y_target.loc[X_tr.index]
            y_te = y_target.loc[X_te.index]
            # Cluster training set and assign test points to nearest centroid
            scaler_tr = StandardScaler()
            X_emb_tr = scaler_tr.fit_transform(X_tr.filter(like='A'))
            clusterer_tr = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusters_tr = clusterer_tr.fit_predict(X_emb_tr)
            # Train experts
            experts_tr = []
            for c in range(n_clusters):
                mask = clusters_tr == c
                if mask.sum() < 10:
                    experts_tr.append(None)
                    continue
                model_c = lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    random_state=random_state,
                    n_jobs=-1
                )
                model_c.fit(X_tr.iloc[mask], y_tr.iloc[mask])
                experts_tr.append(model_c)
            # Predict test set: assign to nearest centroid
            X_emb_te = scaler_tr.transform(X_te.filter(like='A'))
            dists = np.linalg.norm(X_emb_te[:, None, :] - clusterer_tr.cluster_centers_[None, :, :], axis=2)
            closest = dists.argmin(axis=1)
            preds = []
            for i, c in enumerate(closest):
                model_c = experts_tr[c]
                if model_c is None:
                    # fallback to mean of training targets
                    preds.append(y_tr.mean())
                else:
                    preds.append(model_c.predict(X_te.iloc[i:i+1])[0])
            r2 = r2_score(y_te, preds)
            r2_scores.append(r2)
        metrics[target] = {
            'r2_cv_mean': float(np.mean(r2_scores)),
            'r2_cv_std': float(np.std(r2_scores)),
            'rmse': np.nan,  # will compute after fitting full model later
            'mae': np.nan,
            'n_clusters': n_clusters,
            'approach': 'MixtureOfExperts'
        }
    # Fit full models on entire data (store clusterers per target? Use same clusters for simplicity)
    # Already fitted above, but recompute full pipeline for consistency
    full_experts = {}
    for target in y.columns:
        models_c = []
        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() < 10:
                models_c.append(None)
                continue
            X_c = X.iloc[mask]
            y_c = y[target].loc[X_c.index]
            model_c = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1
            )
            model_c.fit(X_c, y_c)
            models_c.append(model_c)
        full_experts[target] = models_c
        # compute in-sample metrics for reference
        preds = []
        for i in range(len(X)):
            c = clusters[i]
            model_c = models_c[c] if models_c[c] is not None else None
            if model_c is None:
                preds.append(y[target].mean())
            else:
                preds.append(model_c.predict(X.iloc[i:i+1])[0])
        y_true = y[target]
        metrics[target]['rmse'] = np.sqrt(mean_squared_error(y_true, preds))
        metrics[target]['mae'] = mean_absolute_error(y_true, preds)
    return {
        'clusterer': clusterer,
        'experts': full_experts,
        'scaler': scaler,
        'metrics': metrics,
        'clusters': clusters
    }

def predict_moe(model_dict, X_new):
    """Predict using Mixture of Experts: assign to nearest cluster centroid."""
    clusterer = model_dict['clusterer']
    scaler = model_dict['scaler']
    experts = model_dict['experts']
    X_emb = X_new.filter(like='A')
    X_emb_scaled = scaler.transform(X_emb)
    dists = np.linalg.norm(X_emb_scaled[:, None, :] - clusterer.cluster_centers_[None, :, :], axis=2)
    closest = dists.argmin(axis=1)
    predictions = {}
    for target, models_c in experts.items():
        preds = []
        for i, c in enumerate(closest):
            model_c = models_c[c]
            if model_c is None:
                preds.append(np.nan)
            else:
                preds.append(model_c.predict(X_new.iloc[i:i+1])[0])
        predictions[target] = np.array(preds)
    return predictions
