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
    
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # הוספת האופציה החדשה לתפריט
    test = st.selectbox("בחר מבחן סטטיסטי:", [
        "📈 ניתוח שיפור (Pre-Post Comparison)",
        "📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)",
        "📉 רגרסיה לינארית (Linear Regression)"
    ])

    st.divider()

    # --- 1. ניתוח שיפור (הכלי החדש שלך) ---
    if "ניתוח שיפור" in test:
        col_setup, col_results = st.columns([1, 2], gap="large")
        
        with col_setup:
            st.subheader("הגדרות הניתוח")
            pre_col = st.selectbox("בחר ציון קדם (Pre):", numeric)
            post_col = st.selectbox("בחר ציון פוסט (Post):", numeric)
            group_col = st.selectbox("בחר משתנה מגמה/קבוצה:", all_cols)
            run_btn = st.button("🚀 הרץ ניתוח שיפור", use_container_width=True)

        with col_results:
            if run_btn:
                # א. יצירת משתנה שיפור (Compute Variable)
                temp_df = df[[pre_col, post_col, group_col]].dropna().copy()
                temp_df['Improvement'] = temp_df[post_col] - temp_df[pre_col]
                
                # ב. חישוב סטטיסטיקה תיאורית לשיפור לפי קבוצה
                stats_summary = temp_df.groupby(group_col)['Improvement'].agg(['mean', 'std', 'count']).reset_index()
                
                st.subheader("📊 ממוצעי שיפור לפי קבוצה")
                st.table(stats_summary.style.format({'mean': "{:.2f}", 'std': "{:.2f}"}))

                # ג. מבחן T או ANOVA להשוואת השיפור בין הקבוצות
                groups = [g['Improvement'].values for _, g in temp_df.groupby(group_col)]
                if len(groups) == 2:
                    t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                    res_text = f"מבחן T להשוואת שיפור: t={t_stat:.3f}, p={p_val:.4f}"
                else:
                    f_stat, p_val = stats.f_oneway(*groups)
                    res_text = f"מבחן ANOVA להשוואת שיפור: F={f_stat:.3f}, p={p_val:.4f}"

                st.info(res_text)

                # ד. גרף שיפור צבעוני (Interaction Plot)
                # נכין את הנתונים לגרף שמראה קו מ-Pre ל-Post
                melted = temp_df.melt(id_vars=[group_col], value_vars=[pre_col, post_col], 
                                      var_name='Time', value_name='Score')
                fig = px.line(melted.groupby([group_col, 'Time'])['Score'].mean().reset_index(), 
                              x='Time', y='Score', color=group_col, markers=True,
                              title="מגמת שיפור: לפני מול אחרי")
                st.plotly_chart(fig, use_container_width=True)

                # שמירה לזיכרון של ה-AI
                st.session_state['last_stat_result'] = f"Improvement Analysis: {res_text}. Mean improvements: {stats_summary.to_dict()}"

    # --- 2. Two-Way ANOVA (הקוד הקודם שלך) ---
    elif "Two-Way ANOVA" in test:
        # ... (הקוד של ה-Two Way ANOVA שהיה לנו קודם) ...
        pass

    # --- ממשק הצ'אט ---
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

    if prompt := st.chat_input("שאל אותי על השיפור..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # שימוש במודל ג'ימיני 2.0
                model = genai.GenerativeModel('gemini-2.0-flash')
                full_context = f"Statistician Expert. Data: {stat_context}. Question: {prompt}. Explain the improvement differences between groups in Hebrew."
                response = model.generate_content(full_context)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"שגיאת AI: {e}")
