"""Approach 1: PCA-reduced Gradient Boosting."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PCA
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
embedding_cols = [f'A{i:02d}' for i in range(64)] + ['A63']
geospatial_cols = ['TH_LAT', 'TH_LONG', 'Elev']
target_cols = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'N', 'K', 'CaCO3']

def run(X_train, X_test, y_train, y_test):
    # Standardize embeddings
    scaler = StandardScaler()
    X_tr_emb = scaler.fit_transform(X_train[embedding_cols])
    X_te_emb = scaler.transform(X_test[embedding_cols])

    # PCA 95% variance
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_tr_emb)
    X_te_pca = pca.transform(X_te_emb)

    # Add geospatial
    X_tr = np.hstack([X_tr_pca, X_train[geospatial_cols].values])
    X_te = np.hstack([X_te_pca, X_test[geospatial_cols].values])

    # Model
    model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
    ))
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # Metrics
    results = {}
    for i, t in enumerate(target_cols):
        results[t] = {
            'R2': r2_score(y_test.iloc[:, i], y_pred[:, i]),
            'RMSE': np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])),
            'MAE': mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
            'n_features': X_tr.shape[1]
        }
    return results, model, pca, scaler
