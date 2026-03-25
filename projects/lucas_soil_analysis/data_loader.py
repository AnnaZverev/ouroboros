import pandas as pd
import numpy as np

def load_lucas_dataset(path):
    """
    Load LUCAS soil dataset with AlphaEarth embeddings.
    Returns: DataFrame (n_samples, n_features)
    """
    df = pd.read_csv(path, low_memory=False)
    # Basic cleaning: strip column names, drop empty columns
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')
    return df

def extract_targets(df):
    """
    Extract nutrient targets: pH_CaCl2, pH_H2O, EC, OC, P, K, CaCO3.
    Returns: target DataFrame
    """
    target_cols = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'K', 'CaCO3']
    # Some columns might have variations; keep only existing ones
    existing = [c for c in target_cols if c in df.columns]
    return df[existing].copy()

def extract_features(df, target_cols):
    """
    Separate feature matrix (embeddings + geospatial) from targets.
    Embeds: columns named A00-A62 (63 dims). Geospatial: lat, lon, elevation, etc.
    """
    # Identify embedding columns: A00 through A62
    embed_cols = [c for c in df.columns if c.startswith('A') and c[1:].isdigit()]
    # Geospatial: any non-target numeric columns that are not embeddings
    exclude = set(target_cols) | set(embed_cols) | {'ID', 'PointID', 'SampleID'}
    geo_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    feature_df = df[embed_cols + geo_cols].copy()
    return feature_df, embed_cols, geo_cols

def prepare_clean_data(df):
    """
    Wrangle dataset:
    - Convert non-numeric to NaN where needed
    - Handle '< LOD' as NaN but track missingness
    - Ensure all feature columns numeric
    Returns: X (features), y (targets), embed_cols, geo_cols
    """
    df = df.copy()
    # Convert any non-numeric strings to NaN across dataframe
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    target_cols = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'K', 'CaCO3']
    y = extract_targets(df)
    X, embed_cols, geo_cols = extract_features(df, target_cols)
    # Ensure no NaNs in X (could impute later)
    return X, y, embed_cols, geo_cols
