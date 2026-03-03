import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
        "📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)",
        "📉 רגרסיה לינארית (Linear Regression)",
        "🔗 מתאם פירסון (Pearson Correlation)",
        "🟣 חי-בריבוע (Chi-Square)"
    ])

    st.divider()
    col_setup, col_results = st.columns([1, 2], gap="large")

    if "One-Way ANOVA" in test:
        with col_setup:
            gv = st.selectbox("משתנה קטגוריאלי (IV):", all_cols)
            dv = st.selectbox("משתנה תלוי (DV):", numeric)
            show_post = st.checkbox("מבחני המשך (Bonferroni)")
            run_btn = st.button("▶️ הרץ ANOVA", use_container_width=True)

        with col_results:
            if run_btn:
                valid_df = df[[gv, dv]].dropna()
                groups = [g[dv].values for _, g in valid_df.groupby(gv)]
                f_stat, p = stats.f_oneway(*groups)
                
                n_valid = len(valid_df)
                df_b, df_w = len(groups)-1, n_valid - len(groups)
                eta2 = (f_stat * df_b) / (f_stat * df_b + df_w)

                st.subheader(f"📊 תוצאות: {significance_badge(p)}")
                apa_table([("F", f"{f_stat:.3f}"), ("df", f"({df_b}, {df_w})"), ("p-value", f"{p:.4f}"), ("η²", f"{eta2:.3f}")])
                
                if p < 0.05 and show_post:
                    st.dataframe(bonferroni_posthoc(valid_df, gv, dv))
                st.plotly_chart(px.box(valid_df, x=gv, y=dv, color=gv, points="all"), use_container_width=True)
                _handle_ai(f"One-Way ANOVA: F({df_b},{df_w})={f_stat:.3f}, p={p:.4f}, η²={eta2:.3f}", "anova")

    elif "Two-Way ANOVA" in test:
        with col_setup:
            f1 = st.selectbox("גורם א' (Factor A):", all_cols, key="f1")
            f2 = st.selectbox("גורם ב' (Factor B):", all_cols, key="f2")
            dv = st.selectbox("משתנה תלוי (DV):", numeric, key="f_dv")
            run_btn = st.button("▶️ הרץ ניתוח דו-כיווני", use_container_width=True)

        with col_results:
            if run_btn:
                valid_df = df[[f1, f2, dv]].dropna()
                model = ols(f"{dv} ~ C({f1}) * C({f2})", data=valid_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                st.subheader("📊 לוח ANOVA (Main Effects & Interaction)")
                st.dataframe(anova_table.style.format("{:.4f}"))
                
                agg_df = valid_df.groupby([f1, f2])[dv].mean().reset_index()
                fig = px.line(agg_df, x=f1, y=dv, color=f2, markers=True, title="Interaction Plot")
                st.plotly_chart(fig, use_container_width=True)
                
                p_inter = anova_table.loc[f"C({f1}):C({f2})", "PR(>F)"]
                _handle_ai(f"Two-Way ANOVA: Interaction p={p_inter:.4f}", "tw_anova")

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
                
                if len(x_vars) > 1:
                    vif_list = []
                    for var in x_vars:
                        x_o = X.drop(columns=[var])
                        r2_v = LinearRegression().fit(x_o, X[var]).score(x_o, X[var])
                        vif = 1 / (1 - r2_v) if r2_v < 1 else 99
                        vif_list.append((var, f"{vif:.2f}"))
                    st.write("**בדיקת מולטי-קוליניאריות (VIF):**")
                    apa_table(vif_list)
                
                _handle_ai(f"Regression: R²={r2:.3f}, Predictors={x_vars}", "reg")

    elif "Correlation" in test:
        with col_setup:
            selected = st.multiselect("בחר משתנים:", numeric, default=numeric[:3] if len(numeric)>2 else numeric)
            run_btn = st.button("▶️ צור מטריצה", use_container_width=True)

        with col_results:
            if run_btn and len(selected) > 1:
                corr_matrix = df[selected].corr()
                st.subheader("🔗 מטריצת מתאמים (Heatmap)")
                fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1,1])
                st.plotly_chart(fig, use_container_width=True)
                _handle_ai(f"Correlation Matrix for: {selected}", "corr")

def _handle_ai(res_str, key):
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1: ctx = st.text_input("הקשר (אופציונלי):", key=f"ctx_{key}")
    with c2:
        st.write("")
        if st.button("🤖 פרש ב-AI", key=f"ai_{key}"):
            store_result(res_str)
            ai_show(f"אתה סטטיסטיקאי. תוצאה: {res_str}. הקשר: {ctx}. נסח APA בעברית, השאר מונחים באנגלית.")
