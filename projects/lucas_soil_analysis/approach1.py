"""Approach 1: Embeddings-Only XGBoost Baseline."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle
from pathlib import Path


class Approach1_EmbeddingsOnly:
    """Predict nutrients using only AlphaEarth embeddings with XGBoost."""

    def __init__(self, n_estimators=200, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}
        self.scaler = None

    def run(self, X, y):
        """
        Run cross-validated training for all nutrients.

        Args:
            X: DataFrame of embedding features (A0-A62)
            y: DataFrame of target nutrients

        Returns:
            results dictionary
        """
        n_samples = len(X)
        n_features = X.shape[1]

        # Standardize embeddings (though XGBoost is insensitive, helps consistency)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # CV setup
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        nutrient_r2 = {}
        all_scores = []

        for nutrient in y.columns:
            y_target = y[nutrient].values

            # Train XGBoost with CV
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

            # Cross-validate
            cv_scores = cross_val_score(model, X_scaled, y_target,
                                        cv=cv, scoring='r2', n_jobs=-1)
            all_scores.extend(cv_scores)

            # Fit on all data for final model
            model.fit(X_scaled, y_target)
            self.models[nutrient] = model

            nutrient_r2[nutrient] = cv_scores.mean()

        results = {
            'n_samples': n_samples,
            'n_features': n_features,
            'cv_scores_all': all_scores,
            'cv_scores_mean': np.mean(all_scores),
            'cv_scores_std': np.std(all_scores),
            'nutrient_r2': nutrient_r2,
            'models': self.models,
            'scaler': self.scaler
        }

        return results

    def save_results(self, path):
        """Save trained models and results to disk."""
        save_dict = {
            'models': self.models,
            'scaler': self.scaler
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':
    # Quick test
    print("Approach1 module loaded successfully.")
