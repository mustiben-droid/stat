import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 Statistic Analyzer")
    
    # ניקוי בסיסי
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # תפריט צד מלא - כל האופציות שביקשת
    analysis_type = st.sidebar.radio("Analysis Menu", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🧪 T-Tests (Independent/Paired)",
        "🛡️ Reliability (Cronbach's Alpha)",
        "🔗 Correlations Matrix",
        "🔲 Frequencies (Chi-Square)"
    ])

    st.divider()
    context_for_ai = ""

    # --- 1. ANOVA (Repeated Measures) ---
    if analysis_type == "📈 ANOVA (Repeated Measures)":
        col_setup, col_out = st.columns([1, 2])
        with col_setup:
            levels = st.multiselect("RM Factors (Levels):", numeric_cols, help="סדר הבחירה קובע את סדר הגרף (Pre אז Post)")
            between_subject = st.selectbox("Between Subjects Factor (Major):", all_cols)
            run_anova = st.button("🚀 Run Full ANOVA")

        if run_anova and len(levels) > 1:
            with col_out:
                try:
                    tdf = df[levels + [between_subject]].dropna().copy()
                    tdf['ID'] = range(len(tdf))
                    # כפיית סדר כרונולוגי לפי הבחירה של המשתמש
                    long_df = pd.melt(tdf, id_vars=['ID', between_subject], value_vars=levels, 
                                     var_name='Time', value_name='Score')
                    long_df['Time'] = pd.Categorical(long_df['Time'], categories=levels, ordered=True)

                    # ANOVA Table Type III
                    model = ols(f'Score ~ C(Time) * C(Q("{between_subject}"))', data=long_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=3)
                    
                    # חישובי SPSS (Mean Square & Eta Squared)
                    anova_table['Mean Square'] = anova_table['sum_sq'] / anova_table['df']
                    ss_res = anova_table.loc['Residual', 'sum_sq']
                    anova_table['η²p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_res)
                    
                    st.write("### Tests of Within-Subjects Effects")
                    st.table(anova_table.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))

                    # גרף - הסדר נשמר לפי הבחירה ב-multiselect
                    st.write("### Descriptives Plot")
                    fig = px.line(long_df.groupby(['Time', between_subject], observed=True)['Score'].mean().reset_index(), 
                                 x='Time', y='Score', color=between_subject, markers=True, template="plotly_white")
                    st.plotly_chart(fig)
                    context_for_ai = f"ANOVA Result: {anova_table.to_dict()}. Target: {between_subject}."
                    st.session_state['last_context'] = context_for_ai
                except Exception as e: st.error(f"Error: {e}")

    # --- 2. RELIABILITY (Cronbach's Alpha) ---
    elif analysis_type == "🛡️ Reliability (Cronbach's Alpha)":
        items = st.multiselect("Select items:", numeric_cols)
        if st.button("Calculate α") and len(items) > 1:
            idat = df[items].dropna()
            k = len(items)
            alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
            st.metric("Cronbach's Alpha", f"{alpha:.3f}")
            context_for_ai = f"Reliability Alpha for {items}: {alpha:.3f}"
            st.session_state['last_context'] = context_for_ai

    # --- 3. CORRELATIONS ---
    elif analysis_type == "🔗 Correlations Matrix":
        vars_corr = st.multiselect("Select variables:", numeric_cols)
        if vars_corr:
            corr = df[vars_corr].corr()
            st.write("### Pearson Correlation Matrix")
            st.table(corr.style.background_gradient(cmap='coolwarm').format("{:.3f}"))
            context_for_ai = f"Correlation matrix: {corr.to_dict()}"
            st.session_state['last_context'] = context_for_ai

    # --- 4. T-TESTS, DESCRIPTIVES, CHI-SQUARE (נשארו כפי שהיו) ---
    # ... (הקוד של T-Test ו-Descriptives נשאר זהה לגרסה הקודמת)

    # --- GEMINI 2.0 FLASH CHAT ---
    if 'last_context' in st.session_state or context_for_ai:
        st.divider()
        st.subheader("🤖 AI Insights (Gemini 2.0 Flash)")
        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        if prompt := st.chat_input("שאל על הממצאים..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    # שימוש במודל 2.0 Flash המדויק
                    model = genai.GenerativeModel('gemini-2.0-flash') 
                    ctx = st.session_state.get('last_context', "General data analysis.")
                    full_p = f"Context: {ctx}\nUser: {prompt}\nAnswer in Hebrew as a statistics expert."
                    resp = model.generate_content(full_p)
                    st.markdown(resp.text)
                    st.session_state.messages.append({"role": "assistant", "content": resp.text})
                except Exception as e: st.error(f"AI Error: {e}")
