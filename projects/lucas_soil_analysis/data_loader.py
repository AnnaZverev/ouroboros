"""Data loading and preprocessing for LUCAS soil dataset."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

embedding_cols = [f'A{i:02d}' for i in range(64)] + ['A63']
geospatial_cols = ['TH_LAT', 'TH_LONG', 'Elev']
target_cols = ['pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'P', 'N', 'K', 'CaCO3']

def load_data(csv_path):
    """Load raw CSV and return cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    # Select relevant columns
    cols_needed = embedding_cols + geospatial_cols + target_cols
    df = df[cols_needed].copy()
    # Replace '< LOD' with NaN and convert to numeric
    for col in target_cols:
        df[col] = pd.to_numeric(df[col].replace('< LOD', np.nan), errors='coerce')
    return df

def prepare_X_y(df):
    """Extract feature matrix X (embeddings + geospatial) and target matrix y."""
    X_emb = df[embedding_cols].values
    X_geo = df[geospatial_cols].values
    X = np.hstack([X_emb, X_geo])
    y = df[target_cols].values
    return X, y, target_cols

def impute_targets(y):
    """Median imputation for missing target values."""
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    y_imp = imp.fit_transform(y)
    return y_imp, imp

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified-like split preserving class balance (none, just random split)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
