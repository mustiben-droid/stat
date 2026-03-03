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
    
    # ניקוי בסיסי
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # אתחול זיכרון גלובלי
    if 'global_context' not in st.session_state:
        st.session_state['global_context'] = "No analysis performed yet."
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # תפריט ניתוחים
    analysis_type = st.sidebar.radio("בחר סוג ניתוח", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🧪 T-Tests",
        "🛡️ Reliability (Cronbach's Alpha)",
        "🔗 Correlations Matrix",
        "🔲 Frequencies (Chi-Square)"
    ])

    st.divider()

    # --- 1. DESCRIPTIVES ---
    if analysis_type == "📊 Descriptives":
        vars_d = st.multiselect("בחר משתנים:", numeric_cols)
        group_d = st.selectbox("קבץ לפי (Major):", ["ללא"] + all_cols)
        if vars_d:
            if group_d == "ללא":
                res = df[vars_d].describe().T
            else:
                res = df.groupby(group_d)[vars_d].describe().stack(level=0)
            st.table(res.style.format("{:.2f}"))
            st.session_state['global_context'] = f"סטטיסטיקה תיאורית למשתנים {vars_d} לפי {group_d}."

    # --- 2. ANOVA (Repeated Measures) ---
    elif analysis_type == "📈 ANOVA (Repeated Measures)":
        col_setup, col_out = st.columns([1, 2])
        with col_setup:
            levels = st.multiselect("רמות זמן (סדר הבחירה קובע):", numeric_cols)
            between = st.selectbox("גורם בין-נבדקי (Major):", all_cols)
            run_anova = st.button("הרץ ANOVA")
        
        if run_anova and len(levels) > 1:
            with col_out:
                try:
                    tdf = df[levels + [between]].dropna().copy()
                    tdf['ID'] = range(len(tdf))
                    # כפיית סדר כרונולוגי כדי שהגרף לא יתהפך
                    long_df = pd.melt(tdf, id_vars=['ID', between], value_vars=levels, var_name='Time', value_name='Score')
                    long_df['Time'] = pd.Categorical(long_df['Time'], categories=levels, ordered=True)
                    
                    model = ols(f'Score ~ C(Time) * C(Q("{between}"))', data=long_df).fit()
                    res = sm.stats.anova_lm(model, typ=3)
                    
                    st.write("### ANOVA Table (Type III)")
                    st.table(res.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))
                    
                    fig = px.line(long_df.groupby(['Time', between], observed=True)['Score'].mean().reset_index(), 
                                 x='Time', y='Score', color=between, markers=True, template="plotly_white")
                    st.plotly_chart(fig)
                    
                    st.session_state['global_context'] = f"ANOVA על {levels} לפי {between}. תוצאות: {res.to_dict()}."
                except Exception as e:
                    st.error(f"שגיאה: {e}")

    # --- 3. T-TESTS ---
    elif analysis_type == "🧪 T-Tests":
        t_mode = st.radio("סוג מבחן", ["Independent (בין קבוצות)", "Paired (אותה קבוצה)"])
        if t_mode == "Independent (בין קבוצות)":
            dv = st.selectbox("משתנה תלוי (ציון):", numeric_cols)
            iv = st.selectbox("משתנה בלתי תלוי (קבוצה):", all_cols)
            if st.button("בצע T-Test"):
                groups = df[iv].unique()
                if len(groups) == 2:
                    g1 = df[df[iv] == groups[0]][dv].dropna()
                    g2 = df[df[iv] == groups[1]][dv].dropna()
                    t_stat, p = stats.ttest_ind(g1, g2)
                    st.metric("p-value", f"{p:.4f}")
                    st.write(f"t-statistic: {t_stat:.3f}")
                    st.session_state['global_context'] = f"T-test בלתי תלוי בין קבוצות {groups} במשתנה {dv}. p={p}."
        else:
            v1 = st.selectbox("זמן 1:", numeric_cols)
            v2 = st.selectbox("זמן 2:", numeric_cols)
            if st.button("בצע Paired T-Test"):
                t_stat, p = stats.ttest_rel(df[v1].dropna(), df[v2].dropna())
                st.metric("p-value", f"{p:.4f}")
                st.session_state['global_context'] = f"T-test מזווג בין {v1} ל-{v2}. p={p}."

    # --- 4. RELIABILITY (קרונבך) ---
    elif analysis_type == "🛡️ Reliability (Cronbach's Alpha)":
        items = st.multiselect("בחר פריטים לשאלון:", numeric_cols)
        if st.button("חשב מהימנות"):
            idat = df[items].dropna()
            k = len(items)
            alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
            st.metric("Cronbach's Alpha (α)", f"{alpha:.3f}")
            st.session_state['global_context'] = f"מהימנות קרונבך ל-{items} יצאה {alpha:.3f}."

    # --- 5. CORRELATIONS ---
    elif analysis_type == "🔗 Correlations Matrix":
        vars_corr = st.multiselect("בחר משתנים למתאם:", numeric_cols)
        if vars_corr:
            corr = df[vars_corr].corr()
            st.write("### Pearson Correlation Matrix")
            st.table(corr.style.background_gradient(cmap='coolwarm').format("{:.3f}"))
            st.session_state['global_context'] = f"מטריצת מתאמים למשתנים {vars_corr}."

    # --- 6. CHI-SQUARE ---
    elif analysis_type == "🔲 Frequencies (Chi-Square)":
        v1 = st.selectbox("משתנה 1 (שורות):", all_cols)
        v2 = st.selectbox("משתנה 2 (עמודות):", all_cols)
        if st.button("הרץ Chi-Square"):
            ct = pd.crosstab(df[v1], df[v2])
            st.table(ct)
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.metric("p-value", f"{p:.4f}")
            st.session_state['global_context'] = f"מבחן Chi-square בין {v1} ל-{v2}. p={p}."

    # --- AI RECOMMENDATIONS & CHAT ---
    st.divider()
    
    col_ai1, col_ai2 = st.columns([1, 1])
    
    with col_ai1:
        st.subheader("💡 יועץ מחקר (Recommendations)")
        if st.button("מה כדאי לי להריץ עכשיו?"):
            with st.spinner("מנתח את המצב..."):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    prompt = f"Context: {st.session_state['global_context']}\nVariables: {all_cols}\nSuggest next steps for a thesis in Hebrew."
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e: st.error(e)

    with col_ai2:
        st.subheader("💬 התייעצות חופשית")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
            
        if prompt := st.chat_input("שאל אותי על התוצאות..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                model = genai.GenerativeModel('gemini-2.0-flash')
                full_p = f"Context: {st.session_state['global_context']}\nUser: {prompt}\nAnswer in Hebrew."
                resp = model.generate_content(full_p)
                st.markdown(resp.text)
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
