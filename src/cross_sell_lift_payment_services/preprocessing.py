from __future__ import annotations

from typing import Dict, Tuple, Any
import pandas as pd 
import numpy as np 
from src.connectors.storage_conn import S3Reader

def _fmt_snake(df: pd.DataFrame) -> pd.DataFrame:
    """Format column names to snake_case"""
    df_copy = df.copy()
    df_copy.columns = [ col.strip().lower()\
                                    .replace(" ", "_")\
                                    .replace("-", "_") \
                        for col in df_copy.columns]
    return df_copy

def _trim_str(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading and trailing whitespace from string columns"""
    df_copy = df.copy()
    for c in df_copy.select_dtypes(include=["object"]).columns:
        df_copy[c] = df_copy[c].str.strip()
        #df[c] = df[c].astype("string").str.strip() # alternative
    return df_copy

def _generic_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary DataFrame of the data"""
    summary = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.tolist(),
        "n_rows": [df.shape[0]],
        "n_cols": [df.shape[1]],
        "missing_values": [df.isnull().sum().tolist()],
        "missing_pct_max": float(df.isna().mean().max()) if len(df) else 0.0,
        "unique_values": [df.nunique().tolist()],
        "unique_pct_max": float(df.nunique().max() / len(df)) if len(df) else 0.0,
        "duplicated_values": [df.duplicated().sum().tolist()],
        "duplicated_pct_max": float(df.duplicated().mean().max()) if len(df) else 0.0,
    }
    return pd.DataFrame(summary)

def _iqr_outlier_bounds(df: pd.Series, col: str, k: float = 1.5) -> Tuple[float, float]:
    """Calculate the IQR outlier bounds for a column"""
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (q1, q3, iqr, lower_bound, upper_bound)

def load_raw_data(raw_data_path: str) -> Dict[str, Any]:
    """Load the raw data from the storage"""

    s3 = S3Reader()
    merchants = s3.read_df(raw_data_path + "merchants.csv")
    cart_events = s3.read_df(raw_data_path + "cart_events.csv")
    cart_incentives = s3.read_df(raw_data_path + "cart_incentives.csv")
    activity_events = s3.read_df(raw_data_path + "activity_events.csv")
    merchants, cart_events, cart_incentives, activity_events = map(_fmt_snake, [merchants, cart_events, cart_incentives, activity_events])
    merchants, cart_events, cart_incentives, activity_events = map(_trim_str, [merchants, cart_events, cart_incentives, activity_events])
    #summary_merchants, summary_cart_sessions, summary_activity_events = map(_generic_summary_df, [merchants, cart_sessions_exposures, activity_events])
    
    return {
        "merchants": merchants,
        "cart_events": cart_events,
        "cart_incentives": cart_incentives,
        "activity_events": activity_events
        #"summary_merchants": summary_merchants,
        #"summary_cart_sessions": summary_cart_sessions,
        #"summary_activity_events": summary_activity_events
    }
    










