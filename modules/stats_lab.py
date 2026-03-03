import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import google.generativeai as genai
from statsmodels.formula.api import ols
import statsmodels.api as sm

from .utils import ai_show, store_result

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 המעבדה הסטטיסטית (StatsMonster Pro)")
    
    # זיהוי עמודות
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # בחירת המבחן - זה ה"מתג" הראשי
    test = st.selectbox("בחר מבחן סטטיסטי מהרשימה:", [
        "📉 ניתוח שיפור (Pre-Post Comparison)",
        "📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)",
        "🔗 רגרסיה לינארית (Linear Regression)"
    ], index=0)

    st.divider()

    # הגדרת עמודות לממשק: ימין להגדרות, שמאל לתוצאות
    col_setup, col_results = st.columns([1, 2], gap="large")

    # --- אפשרות 1: ניתוח שיפור ---
    if "ניתוח שיפור" in test:
        with col_setup:
            st.subheader("⚙️ הגדרות שיפור")
            pre_col = st.selectbox("בחר ציון קדם (Pre):", numeric, key="imp_pre")
            post_col = st.selectbox("בחר ציון פוסט (Post):", numeric, key="imp_post")
            group_col = st.selectbox("בחר משתנה מגמה/קבוצה:", all_cols, key="imp_group")
            run_imp = st.button("🚀 הרץ ניתוח שיפור", use_container_width=True, key="btn_imp")

        with col_results:
            if run_imp:
                try:
                    # חישוב
                    temp_df = df[[pre_col, post_col, group_col]].dropna().copy()
                    temp_df['Improvement'] = temp_df[post_col].values - temp_df[pre_col].values
                    
                    # הצגת טבלה
                    summary = temp_df.groupby(group_col)['Improvement'].agg(['mean', 'std', 'count']).reset_index()
                    st.write("### 📊 ממוצעי שיפור")
                    st.table(summary.style.format({'mean': "{:.2f}", 'std': "{:.2f}"}))
                    
                    # גרף
                    plot_df = temp_df.groupby(group_col)[[pre_col, post_col]].mean().reset_index()
                    plot_df = plot_df.melt(id_vars=group_col, var_name='Time', value_name='Score')
                    fig = px.line(plot_df, x='Time', y='Score', color=group_col, markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state['last_stat_result'] = f"Improvement analysis on {group_col} was performed."
                except Exception as e:
                    st.error(f"שגיאה: {e}")

    # --- אפשרות 2: Two-Way ANOVA ---
    elif "Two-Way ANOVA" in test:
        with col_setup:
            st.subheader("⚙️ הגדרות ANOVA")
            f1 = st.selectbox("גורם א' (Major):", all_cols, key="anova_f1")
            f2 = st.selectbox("גורם ב':", all_cols, key="anova_f2")
            dv = st.selectbox("משתנה תלוי:", numeric, key="anova_dv")
            run_anova = st.button("▶️ הרץ ANOVA", use_container_width=True, key="btn_anova")

        with col_results:
            if run_anova:
                try:
                    formula = f'Q("{dv}") ~ C(Q("{f1}")) * C(Q("{f2}"))'
                    model = ols(formula, data=df).fit()
                    table = sm.stats.anova_lm(model, typ=2)
                    st.write("### 📋 טבלת ANOVA")
                    st.dataframe(table.style.format("{:.4f}"))
                    
                    st.session_state['last_stat_result'] = f"Two-way ANOVA results for {dv} by {f1} and {f2}."
                except Exception as e:
                    st.error(f"שגיאה: {e}")

    # --- חלק הצ'אט (מופיע רק אם יש תוצאות) ---
    if 'last_stat_result' in st.session_state:
        render_chat_interface(st.session_state['last_stat_result'])

def render_chat_interface(stat_context):
    st.divider()
    st.subheader("🤖 צ'אט פירוש תוצאות")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("שאל אותי משהו על הממצאים..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # שימוש במודל 2.0 החדש
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(f"Context: {stat_context}. History: {st.session_state.chat_history}. User: {prompt}")
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"שגיאת AI: {e}")
