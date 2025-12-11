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
    
    # ==================== ADVANCED DATA VALIDATION ====================
    # Detect duplicates, type mismatches, suspicious patterns
    profile['quality_issues'] = _detect_data_quality_issues(df)
    
    return profile


def _detect_data_quality_issues(df: pd.DataFrame) -> dict:
    """
    Detects advanced data quality issues and returns warnings.
    
    Returns:
        Dictionary with warnings and recommendations
    """
    issues = {
        "duplicates": [],
        "type_mismatches": [],
        "suspicious_patterns": [],
        "data_integrity": []
    }
    
    # Check for completely duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        pct = (duplicate_rows / len(df)) * 100
        issues["duplicates"].append({
            "type": "complete_row_duplicates",
            "count": int(duplicate_rows),
            "percentage": round(pct, 2),
            "recommendation": f"Found {duplicate_rows} duplicate rows ({pct:.1f}%). Consider removing with df.drop_duplicates()"
        })
    
    # Check for duplicate values in ID-like columns (should be unique)
    for col in df.columns:
        if any(x in col.lower() for x in ['id', 'key', 'pk', 'unique']):
            if df[col].nunique() < len(df):
                dup_count = len(df) - df[col].nunique()
                issues["duplicates"].append({
                    "type": "column_duplicates",
                    "column": col,
                    "count": dup_count,
                    "recommendation": f"Column '{col}' appears to be an ID but has {dup_count} duplicate values"
                })
    
    # Check for type mismatches (numeric-looking strings, etc.)
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().head(100).astype(str)
            
            # Check if column looks numeric but stored as string
            numeric_like = sum(sample_vals.str.match(r'^-?\d+\.?\d*$'))
            if len(sample_vals) > 0 and numeric_like / len(sample_vals) > 0.8:
                issues["type_mismatches"].append({
                    "column": col,
                    "current_type": "string",
                    "likely_type": "numeric",
                    "recommendation": f"Column '{col}' contains mostly numeric values but stored as text. Convert with pd.to_numeric()"
                })
            
            # Check if column looks boolean but stored as string
            bool_like = sum(sample_vals.str.lower().isin(['true', 'false', 'yes', 'no', '0', '1']))
            if len(sample_vals) > 0 and bool_like / len(sample_vals) > 0.9:
                issues["type_mismatches"].append({
                    "column": col,
                    "current_type": "string",
                    "likely_type": "boolean",
                    "recommendation": f"Column '{col}' appears to be boolean. Convert with df['{col}'].map({{'true': True, 'false': False}})"
                })
    
    # Suspicious patterns
    for col in df.columns:
        col_data = df[col].dropna()
        
        # Check for sparse columns (mostly empty)
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            issues["suspicious_patterns"].append({
                "column": col,
                "issue": "sparse_column",
                "missing_percentage": round(missing_pct, 2),
                "recommendation": f"Column '{col}' is {missing_pct:.1f}% empty. Consider removing if not needed."
            })
        
        # Check for constant columns (all same value)
        if len(col_data) > 0 and col_data.nunique() == 1:
            issues["suspicious_patterns"].append({
                "column": col,
                "issue": "constant_column",
                "value": str(col_data.iloc[0]),
                "recommendation": f"Column '{col}' has only one unique value. Likely not useful for analysis."
            })
        
        # Check for suspiciously high cardinality (likely IDs)
        if col_data.dtype == 'object' and col_data.nunique() / len(df) > 0.95:
            issues["suspicious_patterns"].append({
                "column": col,
                "issue": "high_cardinality",
                "unique_ratio": round(col_data.nunique() / len(df), 3),
                "recommendation": f"Column '{col}' has very high cardinality ({col_data.nunique() / len(df):.1%}). Likely contains IDs or unique identifiers."
            })
    
    # Data integrity checks
    for col in df.columns:
        col_data = df[col].dropna()
        
        # Check for negative values in columns that should be positive
        if any(x in col.lower() for x in ['count', 'quantity', 'amount', 'price', 'revenue', 'age']):
            if df[col].dtype in ['int64', 'float64']:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    issues["data_integrity"].append({
                        "column": col,
                        "issue": "negative_values",
                        "count": int(neg_count),
                        "recommendation": f"Column '{col}' should not have negative values. Found {neg_count} negative entries."
                    })
        
        # Check for unusually large values (likely errors)
        if df[col].dtype in ['int64', 'float64']:
            try:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    extreme_outliers = (z_scores > 5).sum()
                    if extreme_outliers > 0:
                        issues["data_integrity"].append({
                            "column": col,
                            "issue": "extreme_outliers",
                            "count": int(extreme_outliers),
                            "recommendation": f"Column '{col}' has {extreme_outliers} extreme outliers (z-score > 5). Investigate for data entry errors."
                        })
            except Exception:
                pass
    
    # Remove empty categories
    return {k: v for k, v in issues.items() if v}

