"""Approach 3: UMAP-Clustered Specialist Models."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import xgboost as xgb
import pickle
from collections import defaultdict


class Approach3_ClusteredSpecialists:
    """Train separate models per spatial cluster discovered via UMAP."""

    def __init__(self, n_clusters=8, n_estimators=200, max_depth=5, random_state=42):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.cluster_models = defaultdict(dict)  # {cluster_id: {nutrient: model}}
        self.cluster_assignments = None
        self.umap_embedder = None
        self.kmeans = None
        self.scaler = None
        self.feature_names = None

    def run(self, X, y, n_clusters=8):
        """
        Cluster samples and train specialist models per cluster.

        Args:
            X: DataFrame of features (embeddings + geospatial)
            y: DataFrame of targets
            n_clusters: number of spatial clusters

        Returns:
            results dictionary
        """
        n_samples = len(X)
        self.feature_names = X.columns.tolist()
        n_features = len(self.feature_names)

        # Standardize features for clustering
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Dimensionality reduction for clustering (UMAP to 2D)
        self.umap_embedder = umap.UMAP(
            n_components=2,
            random_state=self.random_state,
            n_neighbors=15,
            min_dist=0.1
        )
        X_umap = self.umap_embedder.fit_transform(X_scaled)

        # KMeans clustering in UMAP space
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = self.kmeans.fit_predict(X_umap)
        self.cluster_assignments = cluster_labels

        # Train specialist models per cluster
        cluster_sizes = {}
        cluster_scores = defaultdict(dict)
        all_scores = []

        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            X_cluster = X_scaled[mask]
            y_cluster = y.iloc[mask]

            cluster_sizes[cluster_id] = len(X_cluster)

            # Skip tiny clusters (less than 10 samples) — not enough for CV
            if len(X_cluster) < 10:
                continue

            for nutrient in y.columns:
                y_target = y_cluster[nutrient].values

                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )

                # Cross-validate within cluster
                if len(X_cluster) >= 25:  # enough for 5-fold
                    cv_scores = cross_val_score(model, X_cluster, y_target,
                                                cv=cv, scoring='r2', n_jobs=-1)
                    all_scores.extend(cv_scores)
                    cluster_scores[cluster_id][nutrient] = cv_scores.mean()

                # Fit on all cluster data
                model.fit(X_cluster, y_target)
                self.cluster_models[cluster_id][nutrient] = model

        # Overall metrics: weighted average by cluster size
        total_weighted_r2 = {}
        for nutrient in y.columns:
            weighted_sum = 0
            total_samples = 0
            for cluster_id, scores in cluster_scores.items():
                if nutrient in scores:
                    weighted_sum += scores[nutrient] * cluster_sizes[cluster_id]
                    total_samples += cluster_sizes[cluster_id]
            total_weighted_r2[nutrient] = weighted_sum / total_samples if total_samples > 0 else np.nan

        cv_scores_mean = np.mean(all_scores) if all_scores else np.nan
        cv_scores_std = np.std(all_scores) if len(all_scores) > 1 else 0.0

        results = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'cv_scores_all': all_scores,
            'cv_scores_mean': cv_scores_mean,
            'cv_scores_std': cv_scores_std,
            'nutrient_r2': total_weighted_r2,
            'cluster_models': self.cluster_models,
            'scaler': self.scaler,
            'umap_embedder': self.umap_embedder,
            'kmeans': self.kmeans,
            'feature_names': self.feature_names
        }

        return results

    def predict(self, X):
        """
        Predict using cluster specialists.

        Args:
            X: DataFrame of features

        Returns:
            DataFrame of predictions (n_samples x nutrients)
        """
        X_scaled = self.scaler.transform(X)
        X_umap = self.umap_embedder.transform(X_scaled)
        cluster_labels = self.kmeans.predict(X_umap)

        predictions = {}
        for nutrient in list(self.cluster_models[0].keys()):
            preds = np.zeros(len(X))
            weights = np.zeros(len(X))

            for cluster_id in range(self.n_clusters):
                mask = cluster_labels == cluster_id
                if mask.sum() == 0:
                    continue
                model = self.cluster_models[cluster_id][nutrient]
                cluster_preds = model.predict(X_scaled[mask])
                preds[mask] = cluster_preds
                weights[mask] = self.cluster_sizes.get(cluster_id, 1)

            # Normalize by weights (not strictly needed for point predictions)
            predictions[nutrient] = preds

        return pd.DataFrame(predictions, index=X.index)

    def save_results(self, path):
        """Save cluster models and transformers."""
        save_dict = {
            'cluster_models': dict(self.cluster_models),
            'scaler': self.scaler,
            'umap_embedder': self.umap_embedder,
            'kmeans': self.kmeans,
            'cluster_sizes': self.cluster_sizes,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':
    print("Approach3 module loaded successfully.")
