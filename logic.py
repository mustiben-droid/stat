import scipy.stats as stats
import pandas as pd
import numpy as np

def run_paired_ttest(df, col1, col2):
    clean = df[[col1, col2]].dropna()
    if len(clean) < 2: return None
    t_stat, p_val = stats.ttest_rel(clean[col1], clean[col2])
    d = (clean[col2].mean() - clean[col1].mean()) / clean[col1].std()
    
    # בדיקת נורמליות להפרשים (הנחת יסוד של T-test)
    diff = clean[col2] - clean[col1]
    _ , p_normality = stats.shapiro(diff)
    
    return {
        "t": round(t_stat, 3), "p": round(p_val, 4), "d": round(d, 3),
        "n": len(clean), "mean1": round(clean[col1].mean(), 2),
        "mean2": round(clean[col2].mean(), 2), "sd1": round(clean[col1].std(), 2),
        "sd2": round(clean[col2].std(), 2), "normality_p": round(p_normality, 4)
    }

def run_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method='pearson')

def calculate_cronbach(df, columns):
    df_items = df[columns].dropna()
    k = df_items.shape[1]
    if k < 2: return None
    item_vars = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    return round((k / (k - 1)) * (1 - item_vars / total_var), 3)

def run_linear_regression(df, x_col, y_col):
    """חישוב רגרסיה ליניארית פשוטה"""
    clean = df[[x_col, y_col]].dropna()
    if len(clean) < 3: return None
    
    # חישוב הרגרסיה
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean[x_col], clean[y_col])
    
    return {
        "slope": round(slope, 3),
        "intercept": round(intercept, 3),
        "r_squared": round(r_value**2, 3),
        "p": round(p_value, 4),
        "n": len(clean),
        "x": clean[x_col],
        "y": clean[y_col]
    }