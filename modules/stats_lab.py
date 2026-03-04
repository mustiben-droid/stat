import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 Gemini Stat-Lab Pro: Full Academic Suite")
    
    # ניקוי וזיהוי עמודות
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- תפריט ניתוחים מורחב ---
    analysis_type = st.sidebar.radio("בחר קטגוריית ניתוח:", [
        "📊 Descriptives",
        "📈 ANOVA & Simple Effects",
        "📉 Regressions",
        "🛡️ Reliability (Alpha & Scales)",
        "🔗 Correlations (Pearson/Spearman)",
        "🧪 T-Tests"
    ])

    st.divider()
    col_stats, col_ai = st.columns([0.7, 0.3])

    with col_stats:
        # --- 1. רגרסיות (Regressions) ---
        if analysis_type == "📉 Regressions":
            reg_type = st.selectbox("סוג רגרסיה:", ["Linear Regression", "Logistic Regression"])
            dv = st.selectbox("משתנה תלוי (Y):", numeric_cols)
            ivs = st.multiselect("משתנים בלתי תלויים (X):", numeric_cols + all_cols)
            
            if st.button("Run Regression") and ivs:
                try:
                    X = df[ivs].copy()
                    # טיפול במשתנים קטגוריאליים (כמו Major) באופן אוטומטי
                    X = pd.get_dummies(X, drop_first=True)
                    X = sm.add_constant(X)
                    y = df[dv]
                    
                    if reg_type == "Linear Regression":
                        model = sm.OLS(y, X, missing='drop').fit()
                    else:
                        model = sm.Logit(y, X, missing='drop').fit()
                    
                    st.write(model.summary())
                    st.session_state['last_res'] = f"{reg_type} results: R-squared={model.rsquared if reg_type=='Linear Regression' else 'N/A'}"
                except Exception as e:
                    st.error(f"Error in Regression: {e}")

        # --- 2. מתאמים (Correlations) ---
        elif analysis_type == "🔗 Correlations (Pearson/Spearman)":
            method = st.radio("שיטת מתאם:", ["pearson", "spearman"])
            vars_c = st.multiselect("בחר משתנים למטריצה:", numeric_cols)
            if vars_c:
                corr_matrix = df[vars_c].corr(method=method)
                st.write(f"### {method.capitalize()} Correlation Matrix")
                st.table(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.3f}"))
                st.session_state['last_res'] = f"{method} correlations for {vars_c}"

        # --- 3. אמינות (Reliability) ---
        elif analysis_type == "🛡️ Reliability (Alpha & Scales)":
            items = st.multiselect("Scale Items (מספרי בלבד):", numeric_cols)
            if st.button("Compute Cronbach's Alpha") and len(items) > 1:
                idat = df[items].dropna()
                k = len(items)
                alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
                st.metric("Cronbach's Alpha (α)", f"{alpha:.3f}")
                # הוספת המלצה לפי הערך
                if alpha < 0.7: st.warning("אמינות נמוכה (מתחת ל-0.7)")
                else: st.success("אמינות טובה")
                st.session_state['last_res'] = f"Alpha for {items}: {alpha:.3f}"

        # --- 4. ANOVA & Simple Effects ---
        elif analysis_type == "📈 ANOVA & Simple Effects":
            sub_type = st.radio("תת-ניתוח:", ["Repeated Measures ANOVA", "Simple Main Effects"])
            if sub_type == "Repeated Measures ANOVA":
                levels = st.multiselect("Time Points:", numeric_cols)
                between = st.selectbox("Between-Subject (Major):", all_cols)
                if st.button("Run ANOVA") and len(levels) > 1:
                    tdf = df[levels + [between]].dropna().copy()
                    tdf['ID'] = range(len(tdf))
                    long_df = pd.melt(tdf, id_vars=['ID', between], value_vars=levels, var_name='Time', value_name='Score')
                    model = ols(f'Score ~ C(Time) * C(Q("{between}"))', data=long_df).fit()
                    res = sm.stats.anova_lm(model, typ=3)
                    st.table(res.style.format("{:.3f}"))
            else:
                t_var = st.selectbox("Time Point:", numeric_cols)
                g_var = st.selectbox("Group (Major):", all_cols)
                if st.button("Run Simple Effect"):
                    model = ols(f'Q("{t_var}") ~ C(Q("{g_var}"))', data=df).fit()
                    st.table(sm.stats.anova_lm(model, typ=3).style.format("{:.3f}"))

    # --- טור הצ'אט של ג'ימיני ---
    with col_ai:
        render_ai_sidebar()

def render_ai_sidebar():
    st.subheader("🤖 AI Consultant")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    chat_container = st.container(height=500)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("שאל את ג'ימיני..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container: st.chat_message("user").markdown(prompt)
        
        with chat_container:
            with st.chat_message("assistant"):
                model = genai.GenerativeModel('gemini-2.0-flash')
                ctx = st.session_state.get('last_res', "No analysis results yet.")
                resp = model.generate_content(f"Context: {ctx}\nUser: {prompt}\nAnswer in Hebrew as a stats pro.")
                st.markdown(resp.text)
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
