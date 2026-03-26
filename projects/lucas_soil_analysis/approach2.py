"""Approach 2: Embeddings + Engineered Geospatial Features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
from pathlib import Path


class Approach2_GeospatialFeatures:
    """Add elevation, climate, and land use features to embeddings."""

    def __init__(self, n_estimators=200, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.feature_names = None

    def run(self, X, y):
        """
        Run CV training with combined embeddings + geospatial features.

        Args:
            X: DataFrame with both embeddings and geospatial columns
            y: DataFrame of target nutrients

        Returns:
            results dictionary
        """
        n_samples = len(X)
        self.feature_names = X.columns.tolist()
        n_features = len(self.feature_names)

        # Standardize all features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        nutrient_r2 = {}
        all_scores = []

        for nutrient in y.columns:
            y_target = y[nutrient].values

            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

            cv_scores = cross_val_score(model, X_scaled, y_target,
                                        cv=cv, scoring='r2', n_jobs=-1)
            all_scores.extend(cv_scores)

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
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        return results

    def save_results(self, path):
        """Save models and scaler."""
        save_dict = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':
    print("Approach2 module loaded successfully.")
