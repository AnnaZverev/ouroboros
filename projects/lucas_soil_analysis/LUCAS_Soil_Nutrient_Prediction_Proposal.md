# LUCAS Soil Nutrient Prediction from AlphaEarth Embeddings: Data-Driven Proposal

**Dataset:** `LUCAS_SOIL_2018_AlphaEarth_Embeddings_Wheat.csv`  
**Samples:** 1,556 | **Features:** 64 AlphaEarth embeddings (A00-A63) + geospatial metadata | **Targets:** pH_CaCl2, pH_H2O, EC, OC, P, N, K, CaCO3

**Constraint:** Predictions must use only embeddings and geospatial metadata. No measured nutrients as inputs.

---

## Executive Summary

Exploratory Data Analysis (EDA) reveals that AlphaEarth embeddings capture statistically significant relationships with most soil nutrients. However, the strength and nature of these relationships vary substantially across targets. We propose **three complementary modeling approaches**, each justified by specific findings from the analysis, to achieve the best possible predictive performance.

---

## Key EDA Findings

### 1. Strong Signal in Embeddings for pH and Carbonates

- **pH (both CaCl2 and H2O)** shows the highest linear correlations:
  - Top positive: A06 (0.326/0.353), A41 (0.303/0.331)
  - Top negative: A31 (-0.344/-0.362), A35 (-0.388/-0.413)
- **CaCO3** has the strongest correlation magnitude: A35 (-0.473), A48 (-0.398), A06 (0.369)
- **PCA** reveals that PC1 alone explains much of the variance and correlates strongly with pH and CaCO3 (-0.305 to -0.444). This suggests a dominant latent factor in the embedding space representing acidity/alkalinity.

### 2. Moderate Signal for Nutrients (OC, P, N, K)

- Organic Carbon (OC) and Nitrogen (N) exhibit negative correlations with embeddings A06, A05, A32 (-0.245 to -0.290). They positively load on PC1 (+0.220, +0.255).
- Phosphorus (P) correlates with A26 (+0.200) and A54 (+0.172).
- Potassium (K) shows moderate negative correlations with A31 (-0.210), A58 (-0.203), A41 (-0.188).
- **Mutual Information** confirms these relationships are non-linear as well (MI values up to 0.098 for N with A43).

### 3. Weak Signal for Electrical Conductivity (EC)

- EC's highest absolute correlation is only **0.166** (A37 positive, A15 negative).
- PCA loadings near zero (PC1 = -0.024), indicating EC is nearly orthogonal to the main embedding variance.
- This implies EC is poorly captured by AlphaEarth embeddings and will require either specialized modeling or external data to predict accurately.

### 4. Missing Data Patterns

- CaCO3 is 44% missing (686 of 1556 samples) — must be handled carefully (imputation or separate modeling).
- P has 10% missing (148 samples) after cleaning ('< LOD' values).
- Other targets are complete or nearly so.

---

## Proposed Modeling Approaches

### Approach 1: PCA-Reduced Gradient Boosting (Baseline + Optimal)

**Justification:** PCA shows that the first 27 components explain 95% of embedding variance, and the first 5 PCs already capture 58.3%. Since pH and CaCO3 correlate strongly with PC1, a model on reduced dimensions can denoise and prevent overfitting while preserving signal.

**Implementation:**
- Preprocess: Standardize embeddings → PCA to retain 90-95% variance (≈27 components)
- Model: `LightGBM` or `XGBoost` with optimized hyperparameters
- Prediction: Single multi-output regressor (or separate models per target)
- Expected: Good performance on pH and CaCO3; moderate on others; poor on EC

**Why not more complex?** Simplicity aligns with Principle 5 (Minimalism). This approach tests whether the embedding manifold itself is sufficient.

---

### Approach 2: Target-Specific Mixture-of-Experts

**Justification:** Correlation and MI analyses show **different sets of embeddings** dominate for different targets. A single global model may average out these specializations. EDA reveals natural clusters:
- **Cluster A (pH + CaCO3):** Shared important embeddings (A35, A06, A31, A41, A43) with opposite signs
- **Cluster B (OC, N, P, K):** Moderate signal, distinct patterns (e.g., OC/N negative with A06, P positive with A26)
- **Cluster C (EC):** Weak signal, potentially needs different feature set or model

**Implementation:**
- For each target, select top-k embeddings based on absolute Pearson correlation + MI (e.g., top 10)
- Train specialized model per target (LightGBM or simple MLP)
- Optionally ensemble within clusters
- Uses only embeddings, no external data (stays within constraints)

**Expected:** Improved performance over global model by allowing each target to exploit its own optimal embedding subset.

---

### Approach 3: Hybrid Embeddings + External Geospatial Covariates

**Justification:** EC's poor correlation with embeddings (max |r|=0.166) suggests missing environmental drivers not captured by AlphaEarth's spectral features. To maximize overall accuracy, we can enrich the feature set with **publicly available global raster layers** at the sample coordinates:
- Climate: WorldClim v2.1 (Bio1–Bio19)
- Elevation: SRTM or GMTED2010
- Soil texture: ISRIC SoilGrids (sand/silt/clay fractions)
- Land cover: ESA CCI 300m
- Hydrological features: Distance to rivers/lakes (HydroSHEDS)

These are derived from the **geospatial metadata** (lat/lon) already available, so they comply with the constraint.

**Implementation:**
- Extract raster values at each sample's lat/lon using `rasterio` and available APIs (or pre-downloaded GeoTIFFs)
- Normalize and concatenate with embeddings
- Feature selection (e.g., SHAP) to avoid dilution
- Train model(s) as in Approach 2, now with enriched feature set

**Expected:** Significant boost for EC and possibly nutrients, turning a weak-signal target into a solvable problem.

---

## Comparative Summary

| Approach | Strengths | Weaknesses | Expected Targets Benefit |
|----------|-----------|------------|--------------------------|
| 1. PCA + GBDT | Simple, fast, robust to noise | May lose target-specific embeddings | pH, CaCO3 (good); others (moderate) |
| 2. Mixture-of-Experts | Specialized per target, no external data | More models to tune/maintain | All targets except EC (improved) |
| 3. Hybrid + External | Addresses EC's missing signal, highest ceiling | Requires downloading external rasters, more complex | **EC** (potentially large gain), others (modest) |

---

## Recommended Roadmap

1. **Implement Approach 1** as a quick baseline to establish performance metrics.
2. **Implement Approach 2** in parallel; compare per-target metrics to see which targets benefit most from specialization.
3. **Implement Approach 3** for targets where Approach 2 still shows high error (likely EC). Start with WorldClim climate layers and elevation.
4. **Evaluate** on a hold-out test set (20% stratified). Metrics: R², RMSE, MAE per target.
5. **Select** the best approach per target (ensemble if needed) and produce final prediction pipeline.

---

## Risk Assessment & Mitigation

- **CaCO3 missingness (44%):** Impute using embeddings with a dedicated model before final prediction, or treat as separate task (predicting presence vs quantity).
- **External data download/processing:** May exceed Colab session time. Mitigation: pre-select small area tiles or use APIs that serve point queries (e.g., WorldClim point extractor).
- **Embedding dimensionality:** 65 features is manageable; PCA may discard useful rare signals. Keep full-feature models as alternative.

---

## Conclusion

The EDA confirms that AlphaEarth embeddings contain meaningful information about soil properties, but not uniformly across all targets. By tailoring the modeling strategy to the data-driven findings — PCA for denoised global signal, expert mixtures for target-specific patterns, and external enrichment for weak-signal variables — we can build a robust, high-performance prediction system within the given constraints.

**Next step:** Implement Approach 1 and 2 in Python, generate baseline results, and iterate. I will produce code artifacts and performance reports as deliverables.
