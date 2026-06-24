import numpy as np
import pandas as pd
from scipy import stats

def cramers_v(x: pd.Series, y: pd.Series):
    """Cramér's V + p-value via chi-squared contingency table."""
    ct = pd.crosstab(x, y)
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
    return v, p_val


def detect_type(series: pd.Series) -> str:
    """
    Infer measurement level for a pandas Series:
      'binary'      -> bool dtype OR object/category with exactly 2 unique non-null values
      'numeric'     -> int/float dtype OR coercible to numeric
      'categorical' -> everything else
    """
    if series.dtype == bool or pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # object / category / other
    n_unique = series.dropna().nunique()
    if n_unique <= 2:
        return "binary"

    # try coercing to numeric
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().sum() / max(series.notna().sum(), 1) >= 0.95:
        return "numeric"

    return "categorical"