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
    
    # ניקוי שמות עמודות
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # תפריט ניתוחים
    analysis_type = st.sidebar.radio("Analysis Menu", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🔲 Frequencies (Chi-Square)"
    ])

    st.divider()
    context_for_ai = ""

    # --- REPEATED MEASURES ANOVA ---
    if analysis_type == "📈 ANOVA (Repeated Measures)":
        col_setup, col_out = st.columns([1, 2])
        
        with col_setup:
            # כאן היה התיקון: levels הם מספרים, אבל ה-Between יכול להיות כל עמודה (כמו Major)
            levels = st.multiselect("RM Factors (e.g. Pre, Post):", numeric_cols)
            between_subject = st.selectbox("Between Subjects Factor (Major):", all_cols)
            
            st.subheader("Options")
            display_plots = st.checkbox("Descriptive plots", value=True)
            simple_main = st.checkbox("Simple main effects", value=True)
            run = st.button("🚀 Run Full Analysis")

        with col_out:
            if run and len(levels) > 1:
                try:
                    # הכנת הנתונים
                    temp_df = df[levels + [between_subject]].dropna().copy()
                    temp_df['ID'] = range(len(temp_df))
                    long_df = pd.melt(temp_df, id_vars=['ID', between_subject], value_vars=levels, 
                                     var_name='Time', value_name='Score')
                    
                    # הרצת ANOVA Type III
                    model = ols(f'Score ~ C(Time) * C(Q("{between_subject}"))', data=long_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=3)
                    
                    # חישוב Eta Squared
                    ss_resid = anova_table.loc['Residual', 'sum_sq']
                    anova_table['η²p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_resid)
                    
                    st.write("### Tests of Within-Subjects Effects")
                    st.table(anova_table.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))

                    if display_plots:
                        fig = px.line(long_df.groupby(['Time', between_subject], observed=True)['Score'].mean().reset_index(), 
                                     x='Time', y='Score', color=between_subject, markers=True, template="plotly_white")
                        st.plotly_chart(fig)

                    context_for_ai = f"ANOVA Table: {anova_table.to_dict()}. Target: {between_subject} over {levels}."
                    st.session_state['last_context'] = context_for_ai

                except Exception as e:
                    st.error(f"שגיאה בחישוב: {e}")

    # --- צ'אט עם GEMINI 3 FLASH ---
    if 'last_context' in st.session_state:
        st.divider()
        st.subheader("🤖 Gemini 3 Flash - פרשנות נתונים")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("שאל אותי על הנסיגה או המובהקות..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # שימוש במודל Gemini 3 Flash החדש
                    model = genai.GenerativeModel('gemini-3-flash') 
                    full_prompt = f"Context: {st.session_state['last_context']}\nUser: {prompt}\nענה בעברית כסטטיסטיקאי בכיר."
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except:
                    st.warning("מתחבר למודל גיבוי...")
                    # גיבוי למקרה שהגרסה החדשה טרם הוטמעה ב-Region שלך
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Context: {st.session_state['last_context']}\nUser: {prompt}\nענה בעברית.")
                    st.markdown(response.text)
