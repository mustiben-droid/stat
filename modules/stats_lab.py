import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# ייבוא מה-utils שנמצא בתיקייה שמעלינו
from .utils import (
    ai_show, apa_table, significance_badge, store_result,
    label_eta, label_r, bonferroni_posthoc
)

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 המעבדה הסטטיסטית (StatsMonster Pro)")
    
    numeric = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    test = st.selectbox("בחר מבחן סטטיסטי:", [
        "📈 ניתוח שונות חד-כיווני (One-Way ANOVA)",
        "📉 רגרסיה לינארית (Linear Regression)",
        "🔗 מתאם פירסון (Pearson Correlation)",
        "🟣 חי-בריבוע (Chi-Square)"
    ])

    st.divider()
    col_setup, col_results = st.columns([1, 2], gap="large")

    # --- ANOVA ---
    if "ANOVA" in test:
        with col_setup:
            gv = st.selectbox("משתנה קטגוריאלי (IV):", all_cols)
            dv = st.selectbox("משתנה תלוי (DV):", numeric)
            run_btn = st.button("▶️ הרץ ANOVA", use_container_width=True)

        with col_results:
            if run_btn:
                valid_df = df[[gv, dv]].dropna()
                groups = [g[dv].values for _, g in valid_df.groupby(gv)]
                f_stat, p = stats.f_oneway(*groups)
                
                st.subheader(f"📊 תוצאות: {significance_badge(p)}")
                st.caption(f"N={len(valid_df)} (נופו {len(df)-len(valid_df)} חסרים)")
                
                df_b, df_w = len(groups)-1, len(valid_df)-len(groups)
                eta2 = (f_stat * df_b) / (f_stat * df_b + df_w)
                
                apa_table([("F", f"{f_stat:.3f}"), ("p-value", f"{p:.4f}"), ("η²", f"{eta2:.3f}")])
                st.plotly_chart(px.box(valid_df, x=gv, y=dv, color=gv), use_container_width=True)
                _handle_ai(f"ANOVA: F({df_b},{df_w})={f_stat:.3f}, p={p:.4f}, η²={eta2:.3f}", "anova")

    # --- REGRESSION ---
    elif "Regression" in test:
        with col_setup:
            y_var = st.selectbox("משתנה תלוי (Y):", numeric)
            x_vars = st.multiselect("מנבאים (X):", numeric)
            run_btn = st.button("▶️ הרץ רגרסיה", use_container_width=True)

        with col_results:
            if run_btn and x_vars:
                valid_df = df[x_vars + [y_var]].dropna()
                X, y = valid_df[x_vars], valid_df[y_var]
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                
                st.subheader(f"📊 מודל רגרסיה: R² = {r2:.3f}")
                
                # בדיקת VIF (Multicollinearity)
                if len(x_vars) > 1:
                    st.write("**בדיקת מולטי-קוליניאריות (VIF):**")
                    vif_list = []
                    for var in x_vars:
                        x_o = X.drop(columns=[var])
                        r2_v = LinearRegression().fit(x_o, X[var]).score(x_o, X[var])
                        vif = 1 / (1 - r2_v) if r2_v < 1 else 99
                        vif_list.append((var, f"{vif:.2f}"))
                    apa_table(vif_list)
                
                _handle_ai(f"Regression: R²={r2:.3f}, Predictors={x_vars}", "reg")

# --- Helper ---
def _handle_ai(res_str, key):
    st.divider()
    if st.button("🤖 פרש ב-AI", key=f"ai_{key}"):
        ai_show(f"פרש ב-APA (השאר מונחים באנגלית): {res_str}")
