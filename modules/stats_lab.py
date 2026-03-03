import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 ניתוח מגמות ושיפור (SPSS Expert Mode)")
    
    all_cols = df.columns.tolist()
    numeric = df.select_dtypes(include=["number"]).columns.tolist()

    # שלב 1: הגדרת המשתנים לחישוב
    st.subheader("1️⃣ הגדרת משתני המחקר")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pre_var = st.selectbox("ציון לפני (Pre):", numeric, key="pre")
    with col2:
        post_var = st.selectbox("ציון אחרי (Post):", numeric, key="post")
    with col3:
        group_var = st.selectbox("קבוצת השוואה (למשל: מגמה):", all_cols, key="grp")

    if st.button("🚀 הרץ ניתוח נסיגה ושיפור"):
        # חישובים מאחורי הקלעים
        temp_df = df[[pre_var, post_var, group_var]].dropna().copy()
        temp_df['Diff'] = temp_df[post_var] - temp_df[pre_var]
        
        # קידוד אוטומטי (כמו Recode ב-SPSS)
        def categorize(x):
            if x < 0: return "נסיגה (Decline)"
            elif x > 0: return "שיפור (Improvement)"
            return "ללא שינוי (Stable)"
        
        temp_df['Status'] = temp_df['Diff'].apply(categorize)

        # שלב 2: הצגת התוצאות
        st.divider()
        st.subheader("2️⃣ ממצאים: למי קיימת נסיגה?")
        
        # טבלת שכיחויות (Crosstab)
        ctab = pd.crosstab(temp_df[group_var], temp_df['Status'], margins=True)
        st.write("### טבלת הצלבה (Crosstabulation)")
        st.table(ctab)

        # גרף מגמות
        fig = px.histogram(temp_df, x=group_var, color='Status', barmode='group',
                          title="התפלגות נסיגה ושיפור לפי קבוצה",
                          color_discrete_map={
                              "נסיגה (Decline)": "#EF553B",
                              "שיפור (Improvement)": "#00CC96",
                              "ללא שינוי (Stable)": "#AB63FA"
                          })
        st.plotly_chart(fig)

        # שלב 3: מובהקות סטטיסטית
        chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(temp_df[group_var], temp_df['Status']))
        
        st.subheader("3️⃣ בדיקת מובהקות (Chi-Square Test)")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Chi-Square Value", f"{chi2:.3f}")
        res_col2.metric("Sig. (p-value)", f"{p:.4f}")

        if p < 0.05:
            st.success("✅ נמצא קשר מובהק! סוג המגמה משפיע על הסיכוי לנסיגה או שיפור.")
        else:
            st.info("ℹ️ לא נמצא קשר מובהק. ההבדלים בין הקבוצות בנסיגה/שיפור עשויים להיות מקריים.")

        # שמירה ל-AI
        st.session_state['last_stat_result'] = f"Crosstab of {group_var} by Improvement Status. Chi2={chi2:.2f}, p={p:.4f}"
