import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 Statistic Analyzer & AI Research Partner")
    
    # ניקוי שמות עמודות וזיהוי סוגים
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # משתנים קטגוריאליים (כמו Major, Class, Gender)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # תפריט ניתוחים
    analysis_type = st.sidebar.radio("בחר סוג ניתוח", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🎯 Simple Main Effects",
        "🧪 T-Tests",
        "🛡️ Reliability (Cronbach's Alpha)",
        "🔗 Correlations Matrix",
        "🔲 Frequencies (Chi-Square)"
    ])

    st.divider()

    # אתחול זיכרון AI
    if 'global_context' not in st.session_state:
        st.session_state['global_context'] = "No analysis performed yet."

    # --- 1. DESCRIPTIVES (Major זמין כאן) ---
    if analysis_type == "📊 Descriptives":
        vars_d = st.multiselect("משתנים לניתוח (מספריים):", numeric_cols)
        group_d = st.selectbox("פלח לפי (כאן Major אמור להיות):", ["ללא"] + all_cols)
        if vars_d:
            if group_d == "ללא":
                res = df[vars_d].describe().T
            else:
                res = df.groupby(group_d)[vars_d].describe().stack(level=0)
            st.table(res.style.format("{:.2f}"))
            st.session_state['global_context'] = f"Descriptives for {vars_d} grouped by {group_d}."

    # --- 2. ANOVA (Major זמין כאן) ---
    elif analysis_type == "📈 ANOVA (Repeated Measures)":
        levels = st.multiselect("רמות זמן (מספרי):", numeric_cols)
        between = st.selectbox("גורם בין-נבדקי (Major):", all_cols)
        if st.button("הרץ ANOVA"):
            if len(levels) > 1:
                tdf = df[levels + [between]].dropna().copy()
                tdf['ID'] = range(len(tdf))
                long_df = pd.melt(tdf, id_vars=['ID', between], value_vars=levels, var_name='Time', value_name='Score')
                long_df['Time'] = pd.Categorical(long_df['Time'], categories=levels, ordered=True)
                model = ols(f'Score ~ C(Time) * C(Q("{between}"))', data=long_df).fit()
                res = sm.stats.anova_lm(model, typ=3)
                st.table(res.style.format("{:.3f}"))
                st.session_state['global_context'] = f"ANOVA on {levels} by {between}."

    # --- 3. SIMPLE MAIN EFFECTS (כאן Major קריטי) ---
    elif analysis_type == "🎯 Simple Main Effects":
        target_v = st.selectbox("בחר רמת זמן לבדיקה:", numeric_cols)
        group_v = st.selectbox("השוואה בין קבוצות (Major):", all_cols)
        if st.button("הרץ Simple Main Effect"):
            formula = f'Q("{target_v}") ~ C(Q("{group_v}"))'
            model = ols(formula, data=df).fit()
            res = sm.stats.anova_lm(model, typ=3)
            st.table(res.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))
            st.session_state['global_context'] = f"Simple Main Effect of {group_v} on {target_v}."

    # --- 4. T-TESTS (Major זמין ב-Independent) ---
    elif analysis_type == "🧪 T-Tests":
        t_type = st.radio("סוג מבחן", ["Independent (השוואת Major)", "Paired (לפני/אחרי)"])
        if t_type == "Independent (השוואת Major)":
            dv = st.selectbox("משתנה תלוי:", numeric_cols)
            iv = st.selectbox("משתנה קבוצה (Major):", all_cols)
            if st.button("בצע T-Test"):
                groups = df[iv].unique()
                if len(groups) >= 2:
                    g1 = df[df[iv] == groups[0]][dv].dropna()
                    g2 = df[df[iv] == groups[1]][dv].dropna()
                    t_stat, p = stats.ttest_ind(g1, g2)
                    st.metric("p-value", f"{p:.4f}")
                    st.session_state['global_context'] = f"T-test on {dv} by {iv} ({groups[0]} vs {groups[1]})."
        else:
            v1 = st.selectbox("זמן 1:", numeric_cols)
            v2 = st.selectbox("זמן 2:", numeric_cols)
            if st.button("בצע Paired T-Test"):
                t_stat, p = stats.ttest_rel(df[v1].dropna(), df[v2].dropna())
                st.metric("p-value", f"{p:.4f}")

    # --- 5. RELIABILITY (מספרי בלבד) ---
    elif analysis_type == "🛡️ Reliability (Cronbach's Alpha)":
        items = st.multiselect("בחר פריטים לשאלון:", numeric_cols)
        if st.button("חשב אלפא"):
            idat = df[items].dropna()
            k = len(items)
            alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
            st.metric("Cronbach's Alpha (α)", f"{alpha:.3f}")

    # --- AI RECOMMENDATIONS (Gemini 2.0 Flash) ---
    st.divider()
    if st.button("💡 קבל הצעות להמשך הניתוח"):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Context: {st.session_state['global_context']}\nAll columns: {all_cols}\nProvide research advice in Hebrew."
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e: st.error(e)
