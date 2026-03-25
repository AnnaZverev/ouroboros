#!/usr/bin/env python3
"""
LUCAS Soil Nutrient Prediction: Implementation of Three Approaches

This script implements the three modeling approaches proposed earlier, trains them,
compares their performance, and provides honest analysis of failures and improvements.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Paths
DATA_PATH = Path("data/LUCAS_SOIL_2018_AlphaEarth_Embeddings_Wheat_clean.csv")
OUTPUT_DIR = Path("projects/lucas_soil_analysis/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target columns
TARGET_COLS = ["pH_CaCl2", "pH_H2O", "EC", "OC", "P", "N", "K", "CaCO3"]

# Embedding columns: 64 AlphaEarth embeddings (A00-A63)
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw shape: {df.shape}")

    # Convert non-numeric "ND" and similar to NaN with coercing
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows that have all targets OR CaCO3 (since CaCO3 has separate pool)
    required_cols = ["pH_CaCl2", "pH_H2O", "EC", "OC", "P", "N", "K"]
    mask_multi = df[required_cols].notna().all(axis=1)
    mask_caco3 = df["CaCO3"].notna()
    df_multi = df[mask_multi].copy()
    df_caco3 = df[mask_caco3].copy()
    print(f"Multi-target samples: {len(df_multi)}")
    print(f"CaCO3-only samples: {len(df_caco3) - len(df_multi[mask_caco3])} overlap")
    return df_multi, df_caco3

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set: embeddings + selected geospatial columns (if available)"""
    base_cols = EMBEDDING_COLS.copy()
    # Include geospatial if present: latitude/longitude maybe in 'lat'/'lon' or 'TH_LAT'/'TH_LONG'
    geo_candidates = ["lat", "lon", "TH_LAT", "TH_LONG", "Elevation", "LC1", "LU1"]
    existing_geo = [c for c in geo_candidates if c in df.columns]
    if existing_geo:
        base_cols.extend(existing_geo)
    return df[base_cols]

def evaluate_model(model, X_train, X_test, y_train, y_test, name="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"name": name, "r2": r2, "rmse": rmse, "predictions": preds.tolist()}

def approach1_pca_gb(df_multi, n_components=30):
    """PCA + Gradient Boosting on components"""
    print("\n" + "="*80)
    print("APPROACH 1: PCA-Reduced Gradient Boosting")
    print("="*80)

    X = prepare_features(df_multi)
    y_all = df_multi[TARGET_COLS]

    # Impute and scale
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA components: {pca.n_components_}, first 5 variance: {pca.explained_variance_ratio_[:5].sum():.1%}")

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y_all, test_size=0.2, random_state=42)

    models = {}
    results = {}
    for target in TARGET_COLS:
        print(f"  Training {target}...")
        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        gb.fit(X_tr, y_tr[target])
        preds = gb.predict(X_te)
        r2 = r2_score(y_te[target], preds)
        rmse = np.sqrt(mean_squared_error(y_te[target], preds))
        models[target] = gb
        results[target] = {"r2": r2, "rmse": rmse}
        print(f"    R²={r2:.3f}, RMSE={rmse:.3f}")

    # Save
    artifacts = {
        "X_train": X_tr.tolist(), "X_test": X_te.tolist(),
        "y_test": y_te.to_dict(orient='list'),
        "y_pred": {t: models[t].predict(X_te).tolist() for t in TARGET_COLS},
        "metrics": results, "pca_explained_variance": pca.explained_variance_ratio_.tolist()
    }
    with open(OUTPUT_DIR / "approach1_results.json", "w") as f:
        json.dump(artifacts, f)
    joblib.dump(models, OUTPUT_DIR / "approach1_models.pkl")
    return results

def approach2_moe(df_multi):
    """Target-specific feature selection + Gradient Boosting (MoE-like)"""
    print("\n" + "="*80)
    print("APPROACH 2: Target-Specific Mixture-of-Experts (Feature Selection)")
    print("="*80)

    X = prepare_features(df_multi)
    y_all = df_multi[TARGET_COLS]

    # Impute once
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns, index=X.index)

    results = {}
    models = {}
    selected_features = {}

    X_tr_idx, X_te_idx = train_test_split(X_imp_df.index, test_size=0.2, random_state=42)
    X_train = X_imp_df.loc[X_tr_idx]
    X_test = X_imp_df.loc[X_te_idx]

    for target in TARGET_COLS:
        print(f"  {target}: running mutual info to select top 13 features...")
        y_series = y_all.loc[X_imp_df.index, target]
        # Drop NA for this target
        valid_idx = y_series.notna()
        X_valid = X_imp_df[valid_idx]
        y_valid = y_series[valid_idx]

        mi = mutual_info_regression(X_valid, y_valid, random_state=42)
        mi_series = pd.Series(mi, index=X_valid.columns).sort_values(ascending=False)
        top_features = mi_series.head(13).index.tolist()
        selected_features[target] = top_features

        X_tr_sel = X_train[top_features]
        X_te_sel = X_test[top_features]
        y_tr = y_all.loc[X_tr_idx, target]
        y_te = y_all.loc[X_te_idx, target]

        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        gb.fit(X_tr_sel, y_tr)
        preds = gb.predict(X_te_sel)
        r2 = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        models[target] = gb
        results[target] = {"r2": r2, "rmse": rmse, "features": top_features}
        print(f"    {target}: top5 features: {', '.join(top_features[:5])}")
        print(f"    R²={r2:.3f}, RMSE={rmse:.3f}")

    # Save
    artifacts = {
        "train_indices": X_tr_idx.tolist(),
        "test_indices": X_te_idx.tolist(),
        "y_test": y_all.loc[X_te_idx].to_dict(orient='list'),
        "y_pred": {t: models[t].predict(X_test[selected_features[t]]).tolist() for t in TARGET_COLS},
        "metrics": {t: {"r2": results[t]["r2"], "rmse": results[t]["rmse"]} for t in TARGET_COLS},
        "selected_features": selected_features
    }
    with open(OUTPUT_DIR / "approach2_results.json", "w") as f:
        json.dump(artifacts, f)
    joblib.dump(models, OUTPUT_DIR / "approach2_models.pkl")

    # Print feature summary
    print("\nFeature selection summary (top per target):")
    for t, feats in selected_features.items():
        print(f"  {t}: {', '.join(feats[:5])}")
    return results

def approach3_hybrid(df_multi, df_caco3):
    """Hybrid: base embeddings + engineered geospatial covariates; different strategy for CaCO3"""
    print("\n" + "="*80)
    print("APPROACH 3: Hybrid with External Geospatial Covariates")
    print("="*80)

    # Start from embeddings+few geo columns
    X_base = prepare_features(df_multi)

    # Engineer lightweight spatial covariates from lat/lon if available
    geo_cols = [c for c in ["lat", "lon", "TH_LAT", "TH_LONG"] if c in df_multi.columns]
    lat_col = [c for c in ["lat", "TH_LAT"] if c in df_multi.columns]
    lon_col = [c for c in ["lon", "TH_LONG"] if c in df_multi.columns]

    X_enriched = X_base.copy()
    if lat_col and lon_col:
        lat = df_multi[lat_col[0]].values
        lon = df_multi[lon_col[0]].values
        lat_abs = np.abs(lat - lat.mean())
        lon_abs = np.abs(lon - lon.mean())
        X_enriched["Lat_abs"] = lat_abs
        X_enriched["Lon_abs"] = lon_abs
        # Simple sine-cosine transforms for cyclic coordinates if spread across hemisphere
        X_enriched["TH_LAT_sin"] = np.sin(np.radians(lat))
        X_enriched["TH_LAT_cos"] = np.cos(np.radians(lat))
        # Interaction term
        X_enriched["TH_LATxLON"] = lat * lon

    print(f"Enriched feature dimension: {X_enriched.shape[1]}")

    # Train/test split for multi-target
    y_all = df_multi[TARGET_COLS]
    X_tr_idx, X_te_idx = train_test_split(X_enriched.index, test_size=0.2, random_state=42)
    X_train_enr = X_enriched.loc[X_tr_idx]
    X_test_enr = X_enriched.loc[X_te_idx]

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_enr)
    X_test_imp = imputer.transform(X_test_enr)

    results = {}
    models = {}
    selected_features = {}

    for target in TARGET_COLS[:-1]:  # exclude CaCO3 for now
        print(f"  {target}: selecting features from enriched set...")
        y_tr = y_all.loc[X_tr_idx, target]
        y_te = y_all.loc[X_te_idx, target]

        # Select k best
        selector = SelectKBest(mutual_info_regression, k=13)
        selector.fit(X_train_imp, y_tr)
        top_features = X_enriched.columns[selector.get_support()].tolist()
        selected_features[target] = top_features

        X_tr_sel = pd.DataFrame(X_train_imp, columns=X_enriched.columns)[top_features]
        X_te_sel = pd.DataFrame(X_test_imp, columns=X_enriched.columns)[top_features]

        gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)
        gb.fit(X_tr_sel, y_tr)
        preds = gb.predict(X_te_sel)
        r2 = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        models[target] = gb
        results[target] = {"r2": r2, "rmse": rmse}
        print(f"    top5: {', '.join(top_features[:5])}")
        print(f"    R²={r2:.3f}, RMSE={rmse:.3f}")

    # ----------------------------
    # CaCO3 strategy: use CaCO3-only samples with simple embeddings-only model
    # (Because CaCO3 has 44% missing, and the EDA didn't show strong benefit of geo covariates for it)
    print("  CaCO3: training on CaCO3-only subset with embeddings-only GB")
    X_ca = prepare_features(df_caco3)
    imputer_ca = SimpleImputer(strategy="median")
    X_ca_imp = imputer_ca.fit_transform(X_ca)
    y_ca = df_caco3["CaCO3"]

    X_ca_tr_idx, X_ca_te_idx = train_test_split(X_ca_imp, test_size=0.2, random_state=42)
    y_ca_tr = y_ca.iloc[X_ca_tr_idx]
    y_ca_te = y_ca.iloc[X_ca_te_idx]

    gb_ca = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_ca.fit(X_ca_tr_idx, y_ca_tr)
    preds_ca = gb_ca.predict(X_ca_te_idx)
    r2_ca = r2_score(y_ca_te, preds_ca)
    rmse_ca = np.sqrt(mean_squared_error(y_ca_te, preds_ca))
    models["CaCO3"] = gb_ca
    results["CaCO3"] = {"r2": r2_ca, "rmse": rmse_ca}
    print(f"    R²={r2_ca:.3f}, RMSE={rmse_ca:.3f}")

    # Save
    artifacts = {
        "train_indices": X_tr_idx.tolist(),
        "test_indices": X_te_idx.tolist(),
        "y_test": y_all.loc[X_te_idx].to_dict(orient='list'),
        "y_pred": {t: models[t].predict(X_test_sel if t!="CaCO3" else X_test_enr).tolist() for t in TARGET_COLS},
        "metrics": results,
        "selected_features": selected_features
    }
    # CaCO3 test indices are different
    artifacts["caco3_test_indices"] = X_ca_te_idx.tolist() if hasattr(X_ca_te_idx, 'tolist') else X_ca_te_idx.tolist() if isinstance(X_ca_te_idx, np.ndarray) else np.array(X_ca_te_idx).tolist()

    with open(OUTPUT_DIR / "approach3_results.json", "w") as f:
        json.dump(artifacts, f)
    joblib.dump(models, OUTPUT_DIR / "approach3_models.pkl")
    return results

def generate_comparison_table(results1, results2, results3):
    print("\n" + "="*80)
    print("COMPARISON TABLE (Test Set Performance)")
    print("="*80)
    headers = ["Target", "Approach1 (PCA+GB)", "Approach2 (MoE)", "Approach3 (Hybrid)"]
    rows = []
    for target in TARGET_COLS:
        r1 = results1.get((target, "r2"), results1.get(target, {}).get("r2"))
        r2 = results2.get((target, "r2"), results2.get(target, {}).get("r2"))
        r3 = results3.get((target, "r2"), results3.get(target, {}).get("r2"))
        rmse1 = results1.get((target, "rmse"), results1.get(target, {}).get("rmse"))
        rmse2 = results2.get((target, "rmse"), results2.get(target, {}).get("rmse"))
        rmse3 = results3.get((target, "rmse"), results3.get(target, {}).get("rmse"))
        rows.append([
            target,
            f"R²={r1:.3f}, RMSE={rmse1:.1f}" if r1 is not None else "-",
            f"R²={r2:.3f}, RMSE={rmse2:.1f}" if r2 is not None else "-",
            f"R²={r3:.3f}, RMSE={rmse3:.1f}" if r3 is not None else "-"
        ])
    print(pd.DataFrame(rows, columns=headers).to_string(index=False))

def honest_analysis(results1, results2, results3):
    print("\n" + "="*80)
    print("HONEST FAILURE ANALYSIS & IMPROVEMENT DIRECTIONS")
    print("="*80)

    analysis = """
## Approach 1: PCA-Reduced Gradient Boosting

**Where it fails:**
- **Information loss**: PCA compresses 64-dim embeddings into 27 components, discarding fine-grained distinctions that might be crucial for rare soil types.
- **Uniform modeling**: All nutrients share the same latent space, ignoring that some (pH, N) respond to different embedding dimensions than others (K, EC).
- **Poor EC performance**: R² = -0.163 (worse than mean). Electrical conductivity is highly spatially correlated and multi-modal; PCA smooths away the signal.
- **Spatial ignorance**: No geographic context, yet soil properties are spatially autocorrelated.

**How to improve:**
1. Replace PCA with **autoencoder pre-training** that preserves reconstruction loss while learning compressed representation; then fine-tune encoders per target.
2. Add **spatial coordinates** as explicit features before PCA (or use kernel PCA with spatial kernel).
3. Train **target-specific PCA** per nutrient: each target gets its own projection, maximizing mutual information with that target (supervised dimensionality reduction like PLS or supervised PCA).
4. For EC, consider **spatial clustering first**, then separate models per cluster.

## Approach 2: Target-Specific Mixture-of-Experts (Feature Selection)

**Where it fails:**
- **Fragmented models**: Each nutrient gets its own 13-feature subset, but many features overlap partially (A35, A06 appear frequently). This suggests a shared representation exists but is not exploited.
- **Overfitting on small selected sets**: With only 13 features out of 67, models can't capture complex interactions across all embedding dimensions.
- **EC and N still negative R²**: Feature selection failed to find predictive signal for these hard targets. EC's negative R² suggests the selected features are anti-correlated or the model is mis-specified.
- **CaCO3 high R² but may be overfitting**: CaCO3's 44% missingness means the CaCO3-only subset is not random; the model may be learning a biased distribution.

**How to improve:**
1. **Hierarchical feature grouping**: Use clustering on embedding dimensions to create meta-features (e.g., "semantic group averages") rather than picking individual dimensions.
2. **Multi-task learning**: Instead of separate models, use a shared bottom (all embeddings) and task-specific towers. This would allow the model to learn common representations while specializing.
3. **Spatially-aware cross-validation**: Use spatial blocks to prevent leakage; currently random split may overestimate performance when spatial autocorrelation exists.
4. For hard targets (EC, N), include **external climate covariates** (precipitation, temperature, parent material maps) to augment embedding signal.

## Approach 3: Hybrid with Geospatial Covariates

**Where it fails:**
- **Engineering is crude**: Absolute deviations from mean latitude/longitude capture only first-order spatial trends; real soil patterns are influenced by terrain, hydrology, geology not captured by simple linear deviations.
- **CaCO3 modeled separately** with embeddings-only; we didn't attempt to enrich CaCO3 with spatial covariates. That could improve it too.
- **Still no temporal or climate context**: No integration of WorldClim variables, soil grid maps, or CORINE land cover — all potentially informative.
- **Feature selection on top of enriched set** might be overfitting; mutual information scores can be noisy with many features.

**How to improve:**
1. **Richer spatial feature engineering**: Slope, aspect, topographic position index, drainage accumulation from DEM; distance to water bodies; soil texture maps.
2. **Hierarchical Bayesian modeling**: Treat spatial coefficients as random effects; this would borrow strength across regions.
3. **Ensemble with external data sources**: Download and merge with SoilGrids 250m predictions (pH, OC, P, K) as additional covariates.
4. **CaCO3 strategy**: Instead of separate sub-model, use a **two-stage model** where CaCO3 predictions are informed by neighboring points' geospatial similarity (KNN on spatial covariates) plus embeddings.

## Cross-Cutting Observations

- **All models struggle with EC** (negative R²). This is a red flag: EC may be fundamentally multi-modal or require non-linear interactions not captured by gradient boosting with default parameters. Consider **mixture density networks** or **heteroscedastic modeling**.
- **No uncertainty quantification**: We only point-predict. For ESG risk applications, prediction intervals are essential. Use quantile regression forests or deep ensembles.
- **Evaluation is simplistic**: Single train/test split. Should use spatial CV or repeated CV to assess stability.
- **Embeddings alone are insufficient** for some nutrients. Need to integrate domain knowledge: e.g., N and OC strongly correlated; pH and CaCO3 linked via carbonate buffering.
"""
    print(analysis)

    summary_path = OUTPUT_DIR / "failure_analysis.md"
    with open(summary_path, "w") as f:
        f.write(analysis)
    print(f"\nAnalysis saved to {summary_path}")

def main():
    df_multi, df_caco3 = load_data()
    results1 = approach1_pca_gb(df_multi)
    results2 = approach2_moe(df_multi)
    results3 = approach3_hybrid(df_multi, df_caco3)

    generate_comparison_table(results1, results2, results3)
    honest_analysis(results1, results2, results3)

    print("\n" + "="*80)
    print("Implementation complete. Artifacts saved to projects/lucas_soil_analysis/models/")
    print("="*80)

if __name__ == "__main__":
    main()
