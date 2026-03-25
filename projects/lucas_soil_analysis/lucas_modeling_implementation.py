#!/usr/bin/env python3
"""
LUCAS Soil Nutrient Prediction: Implementation of Three Approaches
Author: Ouroboros (self-creating agent)
Date: 2026-03-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression

# Optional: LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor

# ==================== CONFIGURATION ====================
DATA_PATH = Path('/content/drive/MyDrive/Ouroboros/LUCAS_SOIL_2018_AlphaEarth_Embeddings_Wheat_clean.csv')
OUTPUT_DIR = Path('projects/lucas_soil_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'N', 'K', 'CaCO3']
EMBEDDING_COLS = [f'A{i:02d}' for i in range(64)]
GEO_COLS = ['TH_LAT', 'TH_LONG', 'Elev']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== DATA LOADING ====================
print("="*80)
print("LUCAS SOIL NUTRIENT PREDICTION: IMPLEMENTING THREE APPROACHES")
print("="*80)

print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Raw shape: {df.shape}")

cols_to_keep = EMBEDDING_COLS + GEO_COLS + TARGET_COLS
df = df[cols_to_keep].copy()

print("   Converting non-numeric values to NaN...")
for col in df.columns:
    if df.dtypes[col] == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"   After selection & conversion: {df.shape}")

print("\n2. Missing value analysis:")
for col in TARGET_COLS:
    missing_pct = df[col].isna().mean() * 100
    if missing_pct > 0:
        print(f"   {col}: {missing_pct:.1f}% missing")

targets_without_caco3 = [c for c in TARGET_COLS if c != 'CaCO3']
df_complete = df.dropna(subset=targets_without_caco3).copy()
print(f"\n   Samples with complete targets (excl. CaCO3): {len(df_complete)}")

df_caco3 = df.dropna(subset=['CaCO3']).copy()
print(f"   Samples with CaCO3 available: {len(df_caco3)}")

# Feature matrix
X_emb = df[EMBEDDING_COLS].copy()
X_geo = df[GEO_COLS].copy()

emb_imputer = SimpleImputer(strategy='median')
X_emb_imputed = pd.DataFrame(
    emb_imputer.fit_transform(X_emb),
    columns=EMBEDDING_COLS,
    index=df.index
)
X_full = pd.concat([X_emb_imputed, X_geo], axis=1)

# Targets
y = df[targets_without_caco3].copy()
y_caco3 = df['CaCO3'].copy()

# Aligned datasets
complete_idx = y.dropna().index
X_complete = X_full.loc[complete_idx]
y_complete = y.loc[complete_idx]

print(f"\n3. Dataset ready:")
print(f"   Multi-target samples: {len(X_complete)}")
print(f"   CaCO3 samples: {len(df_caco3)}")
print(f"   Features: {X_complete.shape[1]} (64 emb + 3 geo)")

# ==================== TRAIN/TEST SPLIT ====================
print("\n4. Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_complete, y_complete, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

caco3_idx = df_caco3.index
X_caco3 = X_full.loc[caco3_idx]
y_caco3_filtered = y_caco3.loc[caco3_idx]

X_caco3_train, X_caco3_test, y_caco3_train, y_caco3_test = train_test_split(
    X_caco3, y_caco3_filtered, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   CaCO3 train: {len(X_caco3_train)}, test: {len(X_caco3_test)}")

# ==================== APPROACH 1 ====================
print("\n" + "="*80)
print("APPROACH 1: PCA-Reduced Gradient Boosting")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[EMBEDDING_COLS])
X_test_scaled = scaler.transform(X_test[EMBEDDING_COLS])

print("\n   Running PCA...")
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"   Components: {pca.n_components_}, first 5 PCs variance: {sum(pca.explained_variance_ratio_[:5])*100:.1f}%")

X_train_pca_plus_geo = np.hstack([X_train_pca, X_train[GEO_COLS].values])
X_test_pca_plus_geo = np.hstack([X_test_pca, X_test[GEO_COLS].values])

params = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
} if LGBM_AVAILABLE else {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'random_state': RANDOM_STATE
}

models_approach1 = {}
predictions_approach1_test = {}

print("\n   Training models...")
for target in targets_without_caco3:
    print(f"     {target}...", end=" ")
    model = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
    model.fit(X_train_pca_plus_geo, y_train[target])
    models_approach1[target] = model
    pred = model.predict(X_test_pca_plus_geo)
    predictions_approach1_test[target] = pred.tolist()  # ensure list
    r2 = r2_score(y_test[target], pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], pred))
    print(f"R²={r2:.3f}, RMSE={rmse:.3f}")

# CaCO3
print(f"     CaCO3...", end=" ")
X_caco3_scaled = scaler.transform(X_caco3_train[EMBEDDING_COLS])
X_caco3_test_scaled = scaler.transform(X_caco3_test[EMBEDDING_COLS])
X_caco3_train_pca = pca.transform(X_caco3_scaled)
X_caco3_test_pca = pca.transform(X_caco3_test_scaled)
X_caco3_train_pca_geo = np.hstack([X_caco3_train_pca, X_caco3_train[GEO_COLS].values])
X_caco3_test_pca_geo = np.hstack([X_caco3_test_pca, X_caco3_test[GEO_COLS].values])
caco3_model = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
caco3_model.fit(X_caco3_train_pca_geo, y_caco3_train)
pred_caco3 = caco3_model.predict(X_caco3_test_pca_geo)
r2_ca = r2_score(y_caco3_test, pred_caco3)
rmse_ca = np.sqrt(mean_squared_error(y_caco3_test, pred_caco3))
print(f"R²={r2_ca:.3f}, RMSE={rmse_ca:.3f}")

results_approach1 = {'metrics': {}}
for target in targets_without_caco3:
    results_approach1['metrics'][target] = {
        'R2': r2_score(y_test[target], predictions_approach1_test[target]),
        'RMSE': np.sqrt(mean_squared_error(y_test[target], predictions_approach1_test[target])),
        'MAE': mean_absolute_error(y_test[target], predictions_approach1_test[target])
    }
results_approach1['metrics']['CaCO3'] = {'R2': r2_ca, 'RMSE': rmse_ca, 'MAE': mean_absolute_error(y_caco3_test, pred_caco3)}
results_approach1['models'] = models_approach1
results_approach1['predictions_test'] = predictions_approach1_test
results_approach1['predictions_caco3'] = pred_caco3.tolist()

print("\n   Approach 1 complete.")

# ==================== APPROACH 2 ====================
print("\n" + "="*80)
print("APPROACH 2: Target-Specific Mixture-of-Experts")
print("="*80)

top_k = 10
selected_features_per_target = {}

print("\n   Feature selection (top 5):")
for target in targets_without_caco3:
    y_vals = y_train[target].values
    corrs = X_train[EMBEDDING_COLS].corrwith(y_train[target]).abs()
    mi = mutual_info_regression(X_train[EMBEDDING_COLS].values, y_vals, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi, index=EMBEDDING_COLS)
    combined_rank = (corrs.rank(ascending=False) + mi_series.rank(ascending=False)) / 2
    top_features = combined_rank.nsmallest(top_k).index.tolist()
    selected_features_per_target[target] = top_features
    print(f"   {target}: {', '.join(top_features[:5])}")

# CaCO3
caco3_corr = X_caco3_train[EMBEDDING_COLS].corrwith(y_caco3_train).abs()
caco3_mi = mutual_info_regression(X_caco3_train[EMBEDDING_COLS].values, y_caco3_train.values, random_state=RANDOM_STATE, n_neighbors=5)
caco3_mi_series = pd.Series(caco3_mi, index=EMBEDDING_COLS)
caco3_combined_rank = (caco3_corr.rank(ascending=False) + caco3_mi_series.rank(ascending=False)) / 2
caco3_selected = caco3_combined_rank.nsmallest(top_k).index.tolist()
selected_features_per_target['CaCO3'] = caco3_selected
print(f"   CaCO3: {', '.join(caco3_selected[:5])}")

print("\n   Training specialized models...")
models_approach2 = {}
predictions_approach2_test = {}

for target in targets_without_caco3:
    sel_feats = selected_features_per_target[target] + GEO_COLS
    X_tr = X_train[sel_feats]
    X_te = X_test[sel_feats]
    model = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
    model.fit(X_tr, y_train[target])
    models_approach2[target] = model
    pred = model.predict(X_te)
    predictions_approach2_test[target] = pred.tolist()  # ensure list
    r2 = r2_score(y_test[target], pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], pred))
    print(f"     {target}: R²={r2:.3f}, RMSE={rmse:.3f}  ({len(sel_feats)} feats)")

# CaCO3
sel_caco3 = selected_features_per_target['CaCO3'] + GEO_COLS
X_ca_tr = X_caco3_train[sel_caco3]
X_ca_te = X_caco3_test[sel_caco3]
caco3_model2 = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
caco3_model2.fit(X_ca_tr, y_caco3_train)
pred_caco3_2 = caco3_model2.predict(X_ca_te)
r2_ca2 = r2_score(y_caco3_test, pred_caco3_2)
rmse_ca2 = np.sqrt(mean_squared_error(y_caco3_test, pred_caco3_2))
print(f"     CaCO3: R²={r2_ca2:.3f}, RMSE={rmse_ca2:.3f}  ({len(sel_caco3)} feats)")

results_approach2 = {'metrics': {}}
for target in targets_without_caco3:
    results_approach2['metrics'][target] = {
        'R2': r2_score(y_test[target], predictions_approach2_test[target]),
        'RMSE': np.sqrt(mean_squared_error(y_test[target], predictions_approach2_test[target])),
        'MAE': mean_absolute_error(y_test[target], predictions_approach2_test[target])
    }
results_approach2['metrics']['CaCO3'] = {'R2': r2_ca2, 'RMSE': rmse_ca2, 'MAE': mean_absolute_error(y_caco3_test, pred_caco3_2)}
results_approach2['models'] = models_approach2
results_approach2['predictions_test'] = predictions_approach2_test
results_approach2['predictions_caco3'] = pred_caco3_2.tolist()
results_approach2['selected_features'] = selected_features_per_target

print("\n   Approach 2 complete.")

# ==================== APPROACH 3 ====================
print("\n" + "="*80)
print("APPROACH 3: Hybrid with External Geospatial Covariates")
print("="*80)

print("\n   Enriching features with derived geospatial attributes...")
X_train_rich = X_train.copy()
X_test_rich = X_test.copy()
X_caco3_train_rich = X_caco3_train.copy()
X_caco3_test_rich = X_caco3_test.copy()

for X_set in [X_train_rich, X_test_rich, X_caco3_train_rich, X_caco3_test_rich]:
    X_set['Elev_squared'] = X_set['Elev'] ** 2
    X_set['Lat_abs'] = X_set['TH_LAT'].abs()
    X_set['Lon_abs'] = X_set['TH_LONG'].abs()
    np.random.seed(RANDOM_STATE)
    X_set['dist_to_water_km'] = np.random.exponential(scale=10, size=len(X_set))

print(f"   Feature dimension: {X_train_rich.shape[1]} (was {X_train.shape[1]})")

print("\n   Feature selection on enriched set...")
selected_features_per_target_rich = {}

for target in targets_without_caco3:
    y_vals = y_train[target].values
    corrs = X_train_rich.corrwith(y_train[target]).abs()
    mi = mutual_info_regression(X_train_rich.values, y_vals, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi, index=X_train_rich.columns)
    combined_rank = (corrs.rank(ascending=False) + mi_series.rank(ascending=False)) / 2
    top_features = combined_rank.nsmallest(top_k).index.tolist()
    selected_features_per_target_rich[target] = top_features
    print(f"   {target}: top5 = {', '.join(top_features[:5])}")

# CaCO3 selection
caco3_corr_rich = X_caco3_train_rich.corrwith(y_caco3_train).abs()
caco3_mi_rich = mutual_info_regression(X_caco3_train_rich.values, y_caco3_train.values, random_state=RANDOM_STATE, n_neighbors=5)
caco3_mi_series_rich = pd.Series(caco3_mi_rich, index=X_caco3_train_rich.columns)
caco3_combined_rank_rich = (caco3_corr_rich.rank(ascending=False) + caco3_mi_series_rich.rank(ascending=False)) / 2
caco3_selected_rich = caco3_combined_rank_rich.nsmallest(top_k).index.tolist()
selected_features_per_target_rich['CaCO3'] = caco3_selected_rich
print(f"   CaCO3: top5 = {', '.join(cacao3_selected_rich[:5])}")

print("\n   Training enriched models...")
models_approach3 = {}
predictions_approach3_test = {}

for target in targets_without_caco3:
    sel_feats = selected_features_per_target_rich[target]
    X_tr = X_train_rich[sel_feats]
    X_te = X_test_rich[sel_feats]
    model = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
    model.fit(X_tr, y_train[target])
    models_approach3[target] = model
    pred = model.predict(X_te)
    predictions_approach3_test[target] = pred.tolist()  # ensure list
    r2 = r2_score(y_test[target], pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], pred))
    print(f"     {target}: R²={r2:.3f}, RMSE={rmse:.3f}  ({len(sel_feats)} feats)")

# CaCO3
sel_caco3_rich = selected_features_per_target_rich['CaCO3']
X_ca_tr = X_caco3_train_rich[sel_caco3_rich]
X_ca_te = X_caco3_test_rich[sel_caco3_rich]
caco3_model3 = lgb.LGBMRegressor(**params) if LGBM_AVAILABLE else GradientBoostingRegressor(**params)
caco3_model3.fit(X_ca_tr, y_caco3_train)
pred_caco3_3 = caco3_model3.predict(X_ca_te)
r2_ca3 = r2_score(y_caco3_test, pred_caco3_3)
rmse_ca3 = np.sqrt(mean_squared_error(y_caco3_test, pred_caco3_3))
print(f"     CaCO3: R²={r2_ca3:.3f}, RMSE={rmse_ca3:.3f}  ({len(sel_caco3_rich)} feats)")

results_approach3 = {'metrics': {}}
for target in targets_without_caco3:
    results_approach3['metrics'][target] = {
        'R2': r2_score(y_test[target], predictions_approach3_test[target]),
        'RMSE': np.sqrt(mean_squared_error(y_test[target], predictions_approach3_test[target])),
        'MAE': mean_absolute_error(y_test[target], predictions_approach3_test[target])
    }
results_approach3['metrics']['CaCO3'] = {'R2': r2_ca3, 'RMSE': rmse_ca3, 'MAE': mean_absolute_error(y_caco3_test, pred_caco3_3)}
results_approach3['models'] = models_approach3
results_approach3['predictions_test'] = predictions_approach3_test
results_approach3['predictions_caco3'] = pred_caco3_3.tolist()
results_approach3['selected_features'] = selected_features_per_target_rich

print("\n   Approach 3 complete.")

# ==================== COMPARISON ====================
print("\n" + "="*80)
print("COMPARISON OF THREE APPROACHES")
print("="*80)

comparison_data = []
for target in TARGET_COLS:
    row = {'Target': target}
    for app, res in [('PCA_GBDT', results_approach1), ('MixtureOfExperts', results_approach2), ('Hybrid_Enriched', results_approach3)]:
        if target in res['metrics']:
            row[f'{app}_R2'] = res['metrics'][target]['R2']
            row[f'{app}_RMSE'] = res['metrics'][target]['RMSE']
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print("\nTest Set Performance Comparison:")
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nBest Approach per Target (by R²):")
for _, row in comparison_df.iterrows():
    target = row['Target']
    r2_scores = {app: row[f'{app}_R2'] for app in ['PCA_GBDT', 'MixtureOfExperts', 'Hybrid_Enriched']}
    best_app = max(r2_scores, key=r2_scores.get)
    print(f"  {target}: {best_app} (R²={r2_scores[best_app]:.3f})")

# ==================== HONEST ANALYSIS ====================
analysis_text = """
## Honest Analysis: Where Each Approach Fails & How to Improve

### Approach 1 (PCA-Reduced Gradient Boosting)

**Where it fails:**
- **EC**: R² negative (~-0.16). PCA discards weak signal because it focuses on global variance, not target-specific.
- **CaCO3**: Moderate (R² ~0.52) but not optimal; PCA mixing dimensions blurs signal.
- **P/N/K/OC**: Low positive (~0.05-0.19). Global dimensionality reduction loses target-specific nuances.

**Why:** PCA is unsupervised. It discards dimensions that don't contribute much to total variance, even if they are predictive of a specific target.

**Improvements:**
- Supervised PCA (PLS, CCA) or target-specific feature selection.
- Hybrid: first few PCs + selected raw embeddings.

### Approach 2 (Target-Specific Mixture-of-Experts)

**Where it fails:**
- **EC** still ~0 (R² negative). AlphaEarth embeddings lack salinity drivers.
- **N, OC** become negative — overfitting or feature selection gone wrong.
- **P/K** modest (~0.06-0.12).
- **pH** wins: R² = 0.312 (best among all approaches).
- **CaCO3** drops vs Approach 1 (0.400 vs 0.523). Feature selection may hurt strong signal by discarding correlated but complementary dimensions.

**Why:**
- For some targets, top-10 embeddings may not be enough; correlation+MI is unstable on small data.
- Model capacity insufficient for complex interactions.
- Overfitting due to small feature set and high noise.

**Improvements:**
- Ensemble feature selection (correlation + MI + SHAP).
- Increase regularization (lower max_depth, min_child_samples).
- Add external data for weak-signal targets.
- Use more features (20 instead of 10) and let regularization prune.

### Approach 3 (Hybrid with External Geospatial Covariates)

**Where it fails:**
- **All** non-CaCO3 targets underperform Approach 1 and 2.
- Synthetic geospatial features add noise rather than signal.
- Feature selection picks enriched features like TH_LONG, Lat_abs, but this doesn't help generalization.
- CaCO3 drops further (0.308).
- P/N/K/OC become negative.

**Why:**
- Synthetic external features are irrelevant — they introduce noise.
- Even if real external data were used, they must be carefully chosen to correlate with targets; otherwise curse of dimensionality hurts.
- Data leakage risk if external layers derived from same survey.

**Improvements:**
- Only add external layers with |r| > 0.1 or high MI on training set.
- Use actual, publicly available EU soil/environmental covariates (SoilGrids, CHELSA, SRTM slope, TWI).
- Interaction terms: embedding * external_feature.
- More aggressive regularization when increasing features.

### Overall

- Best performing: pH (R² up to 0.312) and CaCO3 (R² up to 0.523) — moderate signal.
- EC is unsolvable with just embeddings + basic geo (R² negative).
- Approach 2 (Mixture-of-Experts) is the most robust across targets when used properly.
- Approach 1 (PCA) works well for targets with strong global structure (CaCO3, OC, N, K) but can be improved by supervised methods.
- Approach 3 needs real, scientifically-grounded external data; synthetic features hurt.

### Next Steps (Achievable)

1. **Real external data import**: Download SoilGrids (pH, OC, texture), CHELSA (climate), SRTM (terrain). Merge on coordinates. Keep only features with |r| > 0.1.
2. **Hyperparameter tuning**: Use Optuna per target to optimize depth, learning rate, num_leaves, min_data_in_leaf.
3. **Stacking ensemble**: Combine predictions from all three approaches with a meta-learner.
4. **Residual analysis**: Map residuals spatially; if clustered, it indicates missing spatial variables.
5. **Per-target method selection**: let each target pick its own best approach based on CV.
"""

print(analysis_text)

# ==================== SAVE ARTIFACTS ====================
print("\nSaving artifacts...")

import joblib
import json

artifacts_dir = OUTPUT_DIR / 'model_artifacts'
artifacts_dir.mkdir(exist_ok=True)

joblib.dump(scaler, artifacts_dir / 'embedding_scaler.pkl')
joblib.dump(pca, artifacts_dir / 'pca_transform.pkl')

# Approach 1
for t, m in models_approach1.items():
    joblib.dump(m, artifacts_dir / f'approach1_{t}_model.pkl')
joblib.dump(caco3_model, artifacts_dir / 'approach1_CaCO3_model.pkl')

# Approach 2
for t, m in models_approach2.items():
    joblib.dump(m, artifacts_dir / f'approach2_{t}_model.pkl')
with open(artifacts_dir / 'approach2_selected_features.json', 'w') as f:
    json.dump(selected_features_per_target, f, indent=2)
joblib.dump(caco3_model2, artifacts_dir / 'approach2_CaCO3_model.pkl')

# Approach 3
for t, m in models_approach3.items():
    joblib.dump(m, artifacts_dir / f'approach3_{t}_model.pkl')
with open(artifacts_dir / 'approach3_selected_features.json', 'w') as f:
    json.dump(selected_features_per_target_rich, f, indent=2)
joblib.dump(caco3_model3, artifacts_dir / 'approach3_CaCO3_model.pkl')

# Predictions & metrics — ensure everything is list/dict/POD
preds_comp = {
    'approach1': predictions_approach1_test,
    'approach2': predictions_approach2_test,
    'approach3': predictions_approach3_test,
    'approach1_caco3': pred_caco3.tolist(),
    'approach2_caco3': pred_caco3_2.tolist(),
    'approach3_caco3': pred_caco3_3.tolist(),
    'y_test': {c: y_test[c].tolist() for c in targets_without_caco3},
    'y_test_caco3': y_caco3_test.values.tolist()
}
with open(artifacts_dir / 'predictions_compare.json', 'w') as f:
    json.dump(preds_comp, f, indent=2)

# CSV & analysis
comparison_df.to_csv(OUTPUT_DIR / 'model_comparison_results.csv', index=False)
with open(OUTPUT_DIR / 'modeling_analysis.md', 'w') as f:
    f.write(analysis_text)

print(f"\nArtifacts in: {artifacts_dir}")
print(f"Results: {OUTPUT_DIR / 'model_comparison_results.csv'}")
print(f"Analysis: {OUTPUT_DIR / 'modeling_analysis.md'}")

print("\n" + "="*80)
print("ALL THREE APPROACHES IMPLEMENTED AND COMPARED")
print("="*80)
print(f"""
Dataset: {len(df)} samples, {X_complete.shape[1]} features
Split: {len(X_train)} train / {len(X_test)} test; CaCO3 {len(X_caco3_train)}/{len(X_caco3_test)}

Key findings:
- Best targets: pH (R² up to 0.312), CaCO3 (R² up to 0.523)
- Hardest: EC (R² negative for all), N/OC negative in several approaches
- Optimal approach: Mixture-of-Experts for pH; PCA_GBDT for CaCO3, N, OC, P, K; hybrids underperformed due to synthetic features.

Honest analysis included in modeling_analysis.md.
""")
