import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 מעבדת SPSS: ניתוח מגמות ונסיגה")
    
    # ניקוי בסיסי של שמות עמודות
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric = df.select_dtypes(include=["number"]).columns.tolist()

    # תפריט בחירה ראשי
    test_type = st.sidebar.selectbox("בחר סוג ניתוח:", [
        "📈 ניתוח נסיגה ושיפור (מגמות)",
        "📊 סטטיסטיקה תיאורית",
        "🔗 מטריצת מתאמים"
    ])

    st.subheader("1️⃣ הגדרת משתני המחקר")
    
    if "ניתוח נסיגה ושיפור" in test_type:
        col1, col2, col3 = st.columns(3)
        with col1:
            pre_var = st.selectbox("ציון לפני (Pre):", numeric, key="pre_v")
        with col2:
            post_var = st.selectbox("ציון אחרי (Post):", numeric, key="post_v")
        with col3:
            group_var = st.selectbox("קבוצת השוואה (Major):", all_cols, key="grp_v")

        if st.button("🚀 הרץ ניתוח מגמות", use_container_width=True):
            try:
                # בידוד נתונים למניעת שגיאות אינדקס
                temp_df = df[[pre_var, post_var, group_var]].dropna().copy()
                
                # חישוב הפרש בצורה בטוחה (.values מונע ValueError)
                temp_df['Diff'] = temp_df[post_var].values - temp_df[pre_var].values
                
                # קידוד אוטומטי לקטגוריות
                def get_status(x):
                    if x < -0.01: return "נסיגה (Decline)"
                    elif x > 0.01: return "שיפור (Improvement)"
                    return "ללא שינוי (Stable)"
                
                temp_df['Status'] = temp_df['Diff'].apply(get_status)

                # תצוגת טבלת הצלבה (Crosstab)
                st.divider()
                st.subheader("2️⃣ ממצאים: טבלת הצלבה (Crosstabs)")
                ctab = pd.crosstab(temp_df[group_var], temp_df['Status'], margins=True, margins_name="Total")
                st.table(ctab)

                # גרף מגמות
                st.subheader("3️⃣ ייצוג ויזואלי של המגמות")
                fig = px.histogram(temp_df, x=group_var, color='Status', barmode='group',
                                  color_discrete_map={
                                      "נסיגה (Decline)": "#EF553B",
                                      "שיפור (Improvement)": "#00CC96",
                                      "ללא שינוי (Stable)": "#AB63FA"
                                  })
                st.plotly_chart(fig, use_container_width=True)

                # מבחן מובהקות
                chi_data = pd.crosstab(temp_df[group_var], temp_df['Status'])
                chi2, p, dof, expected = stats.chi2_contingency(chi_data)
                
                st.metric("מובהקות סטטיסטית (Sig.)", f"{p:.4f}", 
                          delta="מובהק" if p < 0.05 else "לא מובהק", delta_color="normal")
                
                # שמירת הקשר ל-AI
                st.session_state['last_stat_result'] = (
                    f"Analysis of {group_var} for improvement/decline. "
                    f"Crosstab results: {ctab.to_dict()}. Chi-square p-value: {p:.4f}."
                )

            except Exception as e:
                st.error(f"שגיאה בתהליך החישוב: {e}")

    elif "סטטיסטיקה תיאורית" in test_type:
        vars_to_show = st.multiselect("בחר משתנים:", numeric)
        if vars_to_show:
            st.table(df[vars_to_show].describe().T[['count', 'mean', 'std', 'min', 'max']])

    # --- ממשק הצ'אט המובנה ---
    if 'last_stat_result' in st.session_state:
        render_chat_interface(st.session_state['last_stat_result'])

def render_chat_interface(context):
    st.divider()
    st.subheader("🤖 פרשנות AI לממצאים")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("שאל אותי על הנסיגה או השיפור שמצאנו..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # וודא שהגדרת את ה-API KEY ב-Secrets
                model = genai.GenerativeModel('gemini-1.5-flash')
                full_prompt = f"Context: {context}\nUser Question: {prompt}\nAnswer in Hebrew as a senior statistician."
                response = model.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"שגיאת AI: {e}")
