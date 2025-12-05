import pandas as pd
import numpy as np

def deep_data_profile(df: pd.DataFrame, max_rows: int = 500) -> dict:
    """Fast data profile - optimized for speed with minimal overhead"""
    profile = {}
    
    # Basic shape and columns
    profile['shape'] = tuple(df.shape)  # tuple is faster than list for shape
    profile['columns'] = list(df.columns)
    profile['dtypes'] = df.dtypes.astype(str).to_dict()
    profile['missing'] = df.isnull().sum().to_dict()
    
    # Sample only first 3 rows (faster than 5)
    profile['sample'] = df.head(3).to_dict(orient='records')
    
    # Unique values (fast)
    profile['unique_values'] = {col: int(df[col].nunique()) for col in df.columns}
    
    # Only compute statistics for numeric columns (skip strings)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Describe only numeric columns (limited to reduce computation)
        try:
            profile['describe'] = df[numeric_cols].describe().to_dict()
        except Exception:
            profile['describe'] = {}
    
    # Correlation (only if 2+ numeric columns)
    if len(numeric_cols) > 1:
        try:
            profile['correlation'] = df[numeric_cols].corr().to_dict()
        except Exception:
            profile['correlation'] = {}
    
    # Outliers (only for numeric columns, fast IQR-based detection)
    outliers = {}
    for col in numeric_cols:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
        except Exception:
            pass
    
    if outliers:
        profile['outliers'] = outliers
    
    return profile
