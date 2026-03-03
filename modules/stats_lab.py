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
    st.header("🔬 המעבדה הסטטיסטית")
    
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    test = st.selectbox("בחר מבחן סטטיסטי:", ["📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)"])

    if "Two-Way ANOVA" in test:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            f1 = st.selectbox("גורם א' (Major):", all_cols, key="f1")
            f2 = st.selectbox("גורם ב' (קבוצה):", all_cols, key="f2")
            dv = st.selectbox("משתנה תלוי (ציון):", numeric, key="dv")
            run_btn = st.button("▶️ הרץ ניתוח", use_container_width=True)

        if run_btn:
            valid_df = df[[f1, f2, dv]].dropna()
            
            try:
                # 1. חישוב המודל
                formula = f'Q("{dv}") ~ C(Q("{f1}")) * C(Q("{f2}"))'
                model = ols(formula, data=valid_df).fit()
                anova_res = sm.stats.anova_lm(model, typ=2)

                # 2. יצירת טבלת SPSS עם עמודת P (Sig.)
                spss = anova_res.copy()
                spss['Mean Square'] = spss['sum_sq'] / spss['df']
                spss = spss[['sum_sq', 'df', 'Mean Square', 'F', 'PR(>F)']]
                spss.columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig. (P-value)']
                
                # ניקוי שמות השורות
                new_idx = [i.replace('C(Q("', '').replace('"))', '').replace(':', ' x ') for i in spss.index]
                spss.index = new_idx

                st.subheader("📋 טבלת ANOVA בסגנון SPSS")
                st.table(spss.style.format({'Sig. (P-value)': "{:.4f}", 'F': "{:.3f}", 'Mean Square': "{:.3f}"}))

                # 3. הגרף הצבעוני שרצית
                st.subheader("📈 תרשים התפלגות קבוצתית")
                fig = px.box(valid_df, x=f1, y=dv, color=f2, points="all", 
                             title=f"התפלגות {dv} לפי {f1} ו-{f2}",
                             color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig, use_container_width=True)

                # שמירת התוצאה לזיכרון עבור הצ'אט
                summary_res = f"ANOVA results for {dv}: " + ", ".join([f"{k}: p={v:.4f}" for k,v in spss['Sig. (P-value)'].dropna().items()])
                st.session_state['last_stat_result'] = summary_res

            except Exception as e:
                st.error(f"שגיאה בחישוב: {e}")

    # --- חלק הצ'אט הרציף (תמיד מופיע בתחתית אחרי הרצה) ---
    if 'last_stat_result' in st.session_state:
        render_chat_interface(st.session_state['last_stat_result'])

def render_chat_interface(stat_context):
    st.divider()
    st.subheader("🤖 צ'אט ייעוץ על הממצאים")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # הצגת היסטוריה
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # קלט מהמשתמש
    if prompt := st.chat_input("שאל אותי על התוצאות (למשל: למה ה-P לא מובהק?)"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                full_context = f"הסטודנט ניתח נתונים. תוצאות ה-ANOVA: {stat_context}. היסטוריית שיחה: {st.session_state.chat_history}. שאלה נוכחית: {prompt}"
                response = model.generate_content(full_context)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"שגיאה בצ'אט: {e}")
