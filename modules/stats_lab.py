import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import google.generativeai as genai
from statsmodels.formula.api import ols
import statsmodels.api as sm

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 מעבדת SPSS מתקדמת")
    
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    test = st.selectbox("בחר ניתוח (Analyze):", [
        "📈 ניתוח שיפור (Pre-Post Comparison)",
        "📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)",
        "🧪 מבחן t למדגמים בלתי תלויים (Independent t-test)",
        "🧪 מבחן t למדגמים מזווגים (Paired t-test)",
        "🔗 מתאם פירסון (Correlations)",
        "📉 רגרסיה לינארית (Linear Regression)",
        "🛡️ בדיקת מהימנות (Cronbach's Alpha)",
        "📊 סטטיסטיקה תיאורית (Descriptives)"
    ])

    st.divider()
    col_setup, col_results = st.columns([1, 2], gap="large")
    analysis_summary = ""

    # --- 1. Independent t-test (SPSS Format) ---
    if "Independent t-test" in test:
        with col_setup:
            dv = st.selectbox("Test Variable (רציף):", numeric)
            iv = st.selectbox("Grouping Variable (קבוצות):", all_cols)
            run = st.button("הרץ t-test")
        with col_results:
            if run:
                grps = df.groupby(iv)[dv]
                if len(grps) == 2:
                    g_names = list(grps.groups.keys())
                    t_stat, p = stats.ttest_ind(grps.get_group(g_names[0]).dropna(), grps.get_group(g_names[1]).dropna())
                    df_val = len(df[dv].dropna()) - 2
                    
                    # טבלת SPSS: Independent Samples Test
                    t_res = pd.DataFrame({
                        "t": [t_stat], "df": [df_val], "Sig. (2-tailed)": [p],
                        "Mean Difference": [grps.get_group(g_names[0]).mean() - grps.get_group(g_names[1]).mean()]
                    }, index=["Equal variances assumed"])
                    
                    st.write("### Independent Samples Test")
                    st.table(t_res.style.format("{:.3f}"))
                    analysis_summary = f"t-test between {g_names}: t={t_stat:.2f}, p={p:.4f}"
                else: st.error("המשתנה חייב להכיל 2 קבוצות בדיוק (למשל: בנים/בנות)")

    # --- 2. Correlations (SPSS Format) ---
    elif "Correlations" in test:
        with col_setup:
            vars_corr = st.multiselect("בחר משתנים למטריצה:", numeric)
            run = st.button("צור מטריצה")
        with col_results:
            if run and len(vars_corr) > 1:
                corr_matrix = df[vars_corr].corr()
                st.write("### Correlations")
                st.table(corr_matrix.style.format("{:.3f}").background_gradient(cmap='coolwarm', axis=None))
                analysis_summary = f"Pearson Correlations for {vars_corr}"

    # --- 3. Linear Regression (SPSS Format) ---
    elif "Regression" in test:
        with col_setup:
            y = st.selectbox("Dependent Variable:", numeric)
            x = st.multiselect("Independent(s):", numeric)
            run = st.button("הרץ רגרסיה")
        with col_results:
            if run and x:
                X = sm.add_constant(df[x])
                model = sm.OLS(df[y], X, missing='drop').fit()
                
                # טבלת Coefficients (סגנון SPSS)
                coef_df = pd.DataFrame({
                    "B": model.params,
                    "Std. Error": model.bse,
                    "t": model.tvalues,
                    "Sig.": model.pvalues
                })
                st.write("### Coefficients")
                st.table(coef_df.style.format("{:.3f}"))
                st.write(f"**Model Summary: R Square = {model.rsquared:.3f}**")
                analysis_summary = f"Regression on {y}: R2={model.rsquared:.3f}"

    # --- 4. Reliability (SPSS Format) ---
    elif "Reliability" in test:
        with col_setup:
            items = st.multiselect("בחר פריטים:", numeric)
            run = st.button("חשב Alpha")
        with col_results:
            if run and len(items) > 1:
                idat = df[items].dropna()
                alpha = (len(items)/(len(items)-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
                
                # טבלת SPSS: Reliability Statistics
                rel_tab = pd.DataFrame({"Cronbach's Alpha": [alpha], "N of Items": [len(items)]})
                st.write("### Reliability Statistics")
                st.table(rel_tab.style.format({"Cronbach's Alpha": "{:.3f}", "N of Items": "{:.0f}"}))
                analysis_summary = f"Reliability: Alpha={alpha:.3f}"

    # --- 5. Two-Way ANOVA (SPSS Format) ---
    elif "Two-Way ANOVA" in test:
        with col_setup:
            f1, f2, dv = st.selectbox("גורם 1:", all_cols), st.selectbox("גורם 2:", all_cols), st.selectbox("תלוי:", numeric)
            run = st.button("הרץ ANOVA")
        with col_results:
            if run:
                model = ols(f'Q("{dv}") ~ C(Q("{f1}")) * C(Q("{f2}"))', data=df).fit()
                res = sm.stats.anova_lm(model, typ=2)
                res['Mean Square'] = res['sum_sq'] / res['df']
                res = res[['sum_sq', 'df', 'Mean Square', 'F', 'PR(>F)']]
                res.columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.']
                st.write("### Tests of Between-Subjects Effects")
                st.table(res.style.format("{:.3f}"))
                analysis_summary = f"ANOVA for {dv} by {f1},{f2}"

    # --- צ'אט ---
    if analysis_summary:
        st.session_state['last_stat_result'] = analysis_summary
    if 'last_stat_result' in st.session_state:
        render_chat_interface(st.session_state['last_stat_result'])

def render_chat_interface(context):
    st.divider()
    st.subheader("🤖 צ'אט פירוש ממצאים")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("שאל על התוצאות..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                resp = model.generate_content(f"Analyze Context: {context}. User: {p}. Answer in Hebrew.")
                st.markdown(resp.text)
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
            except Exception as e: st.error(f"AI Error: {e}")
