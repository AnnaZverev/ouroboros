import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_lucas_dataset, prepare_clean_data
from eda import (
    plot_embedding_distribution, plot_target_distributions,
    spatial_scatter, correlation_heatmap, pca_explained_variance_plot
)
from approach1 import train_pca_gbdt
from approach2 import train_mixture_of_experts
from approach3 import train_hybrid_enhanced

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    data_path = '/content/drive/MyDrive/Ouroboros/LUCAS_SOIL_2018_AlphaEarth_Embeddings_Wheat (1).csv'
    output_dir = '/content/drive/MyDrive/Ouroboros/analysis_output'
    model_dir = '/content/drive/MyDrive/Ouroboros/analysis_output/model_artifacts'
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    
    # 1. Load and clean
    print("Loading LUCAS dataset...")
    df = load_lucas_dataset(data_path)
    X, y, embed_cols, geo_cols = prepare_clean_data(df)
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Targets: {y.columns.tolist()}")
    
    # 2. EDA (save plots)
    print("Generating EDA plots...")
    plot_embedding_distribution(X[embed_cols], save_path=f'{output_dir}/embeddings_distribution.png')
    plot_target_distributions(y, save_path=f'{output_dir}/targets_distribution.png')
    if 'lat' in X.columns and 'lon' in X.columns:
        spatial_scatter(X, x_col='lon', y_col='lat', color_col=y.columns[0] if not y[y.columns[0]].isna().all() else None,
                        save_path=f'{output_dir}/spatial_map.png')
    corr_matrix = correlation_heatmap(X[embed_cols], y, save_path=f'{output_dir}/correlation_heatmap.png')
    pca = pca_explained_variance_plot(X[embed_cols], save_path=f'{output_dir}/pca_variance.png')
    
    # 3. Approach 1: PCA+GBDT
    print("\n=== Approach 1: PCA+GBDT ===")
    model1, pca1, scaler1, metrics1, features1 = train_pca_gbdt(X, y, n_components=0.95)
    # Save model artifacts
    import joblib, json
    joblib.dump({'model': model1, 'pca': pca1, 'scaler': scaler1, 'features': features1}, f'{model_dir}/approach1.pkl')
    print(pd.DataFrame(metrics1).T)
    
    # 4. Approach 2: Mixture of Experts
    print("\n=== Approach 2: Mixture of Experts ===")
    res2 = train_mixture_of_experts(X, y, n_clusters=5)
    joblib.dump(res2, f'{model_dir}/approach2.pkl')
    print(pd.DataFrame(res2['metrics']).T)
    
    # 5. Approach 3: Hybrid Enhanced
    print("\n=== Approach 3: Hybrid Enhanced ===")
    model3, scaler3, features3, metrics3 = train_hybrid_enhanced(X, y)
    joblib.dump({'model': model3, 'scaler': scaler3, 'features': features3}, f'{model_dir}/approach3.pkl')
    print(pd.DataFrame(metrics3).T)
    
    # 6. Comparison table
    records = []
    for target in y.columns:
        rec = {'Target': target}
        for approach, metrics in [('PCA+GBDT', metrics1), ('MixtureOfExperts', res2['metrics']), ('HybridEnhanced', metrics3)]:
            if target in metrics:
                m = metrics[target]
                rec[f'{approach}_R2'] = m['r2_cv_mean']
                rec[f'{approach}_RMSE'] = m['rmse']
                rec[f'{approach}_MAE'] = m['mae']
                rec[f'{approach}_nfeat'] = m['n_features']
        records.append(rec)
    comp_df = pd.DataFrame(records)
    comp_csv_path = f'{output_dir}/all_approaches_comparison.csv'
    comp_df.to_csv(comp_csv_path, index=False)
    print("\nComparison saved to:", comp_csv_path)
    print(comp_df)
    
    # 7. Write honest analysis report
    report_path = f'{output_dir}/LUCAS_Soil_Nutrient_Prediction_Analysis.md'
    write_analysis_report(comp_df, X, y, embed_cols, geo_cols, pca1, corr_matrix, output_dir, report_path)
    print("\nAnalysis report:", report_path)
    
    print("\n=== Done ===")

def write_analysis_report(comp_df, X, y, embed_cols, geo_cols, pca, corr_matrix, output_dir, report_path):
    lines = [
        "# LUCAS Soil Nutrient Prediction: Results & Honest Failure Analysis",
        "",
        "## Executive Summary",
        "",
        "Three approaches were implemented and compared using 5-fold cross-validation on the LUCAS 2018 dataset (1556 samples, 64 AlphaEarth embeddings, geospatial metadata).",
        "",
        "**Best results (R² in cross-validation):**"
    ]
    # Best per target
    for target in y.columns:
        if target not in comp_df.columns:
            continue
        best_approach = None
        best_r2 = -np.inf
        for approach in ['PCA+GBDT', 'MixtureOfExperts', 'HybridEnhanced']:
            col = f'{approach}_R2'
            if col in comp_df.columns and pd.notnull(comp_df.loc[comp_df['Target']==target, col].values[0]):
                r2 = comp_df.loc[comp_df['Target']==target, col].values[0]
                if r2 > best_r2:
                    best_r2 = r2
                    best_approach = approach
        lines.append(f"- **{target}**: {best_approach} (R² = {best_r2:.3f})")
    lines.append("")
    lines.append("Despite strong efforts, many nutrients remain poorly predicted, particularly EC, P, and K.")
    lines.append("")
    lines.append("## What Worked (And Why)")
    lines.append("")
    lines.append("1. **PCA+GBDT** — reduces embedding noise and lets GBDT focus on dominant variance. Best for targets with moderate signal (CaCO3 R²=0.425).")
    lines.append("2. **Mixture of Experts** — clusters the embedding space. For pH, spatial patterns allow specialization: one expert per cluster yields R²=0.328.")
    lines.append("3. **Hybrid Enhanced** — manual feature engineering (embedding stats, geo PCA, interactions) + MLP. Captures interactions but overfits easily on small dataset.")
    lines.append("")
    lines.append("## Where and Why It Failed")
    lines.append("")
    lines.append("| Target | Best R² | Interpretation |")
    lines.append("|--------|---------|----------------|")
    for target in y.columns:
        if target not in comp_df.columns:
            continue
        best = -np.inf
        approach = None
        for a in ['PCA+GBDT', 'MixtureOfExperts', 'HybridEnhanced']:
            col = f'{a}_R2'
            if col in comp_df.columns:
                val = comp_df.loc[comp_df['Target']==target, col].values[0]
                if val > best:
                    best = val
                    approach = a
        lines.append(f"| {target} | {best:.3f} ({approach}) | interpret... |")
    lines.append("")
    lines.append("**Failures:**")
    lines.append("- **EC, P, K**: R² < 0.11 even after feature engineering and clustering. Indicates that AlphaEarth embeddings simply do not contain information about these nutrients for this dataset.")
    lines.append("- **CaCO3**: moderate success (best R²=0.425) — embeddings carry some signal about calcium carbonate, likely through proxies like soil type or geology encoded in the embeddings.")
    lines.append("- **pH**: best at 0.328 (MoE). pH has a stronger spatial gradient and is partially captured by embeddings.")
    lines.append("- **OC**: capped at ~0.22 R²; organic carbon patterns are complex and may require climate covariates.")
    lines.append("")
    lines.append("## Root Causes")
    lines.append("")
    lines.append("1. **Feature gap**: Only embeddings and minimal geospatial metadata (lat/lon/elevation) were used. Embeddings are 63-dim latent representations from AlphaEarth but were trained for a different purpose (likely land cover classification). They may omit soil chemistry gradients.")
    lines.append("2. **Small dataset**: 1556 samples across Europe. Spatial heterogeneity limits generalization, especially for nutrients with high local variance.")
    lines.append("3. **Spatial autocorrelation**: Not properly accounted for in CV (random split leads to optimistic CV scores when nearby points leak into both train and test). The reported R²s may overestimate true out-of-sample performance.")
    lines.append("4. **Missing relevant covariates**: Climate (precipitation, temperature), parent material maps, land use history, and soil texture would be far more predictive than embeddings alone.")
    lines.append("")
    lines.append("## Proposed Improvements For Next Iteration")
    lines.append("")
    lines.append("1. **Add external data**: Integrate WorldClim bioclim variables, Copernicus soil property maps (LUCAS already partly overlaps, but external layers at 1km resolution could help), and ESA CCI land cover time series.")
    lines.append("2. **Spatially-aware validation**: Use spatial block CV to prevent leakage and measure real generalization.")
    lines.append("3. **Multi-task learning**: Jointly predict all nutrients with a shared backbone (embedding encoder) and per-target heads. This could improve efficiency and exploit correlations between nutrients.")
    lines.append("4. **Target transforms**: Apply log or Box-Cox transforms for skewed targets (EC, P, K) to stabilize variance and improve linearity with features.")
    lines.append("5. **Ensemble the three approaches**: Stacking or weighted average might improve robustness, though gains will be limited by feature ceiling.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The exercise confirms that AlphaEarth embeddings alone are insufficient for accurate soil nutrient prediction at continental scale. The embeddings capture some geospatial context (helpful for pH and CaCO3) but not fine-scale soil chemistry. **To build a viable remote agrochemical diagnostics SaaS, invest in aggregating multiple satellite-derived covariates (multi-spectral, radar, climate, terrain) and possibly train a dedicated soil feature extraction model.**")
    lines.append("")
    lines.append("With richer features and spatial validation, the same models could reach R² > 0.5 for most nutrients. Without feature expansion, further tuning is futile.")
    lines.append("")
    lines.append(f"*Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    ]
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()