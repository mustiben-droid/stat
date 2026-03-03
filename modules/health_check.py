import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats

def render_health_check(df: pd.DataFrame):
    st.subheader("🏥 בדיקת בריאות הנתונים (Data Health Check)")
    st.write("לפני שרצים לניתוחים, בוא נוודא שהנתונים שלך תקינים למחקר.")

    # 1. סקירת ערכים חסרים
    st.markdown("### 🔍 ערכים חסרים (Missing Values)")
    missing_data = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'ערכים חסרים': missing_data,
        'אחוז חסר': missing_pct
    }).filter(items=df.columns, axis=0)
    
    # הצגת עמודות עם חוסרים בלבד
    only_missing = missing_df[missing_df['ערכים חסרים'] > 0]
    
    if not only_missing.empty:
        st.warning(f"נמצאו ערכים חסרים ב-{len(only_missing)} משתנים.")
        st.dataframe(only_missing.style.format({'אחוז חסר': "{:.1f}%"}))
    else:
        st.success("מעולה! אין ערכים חסרים בבסיס הנתונים.")

    st.divider()

    # 2. בדיקת התפלגות נורמלית (למשתנים מספריים)
    st.markdown("### 📊 בדיקת נורמליות (Normality Test)")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        target_col = st.selectbox("בחר משתנה לבדיקת התפלגות:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # היסטוגרמה עם קו התפלגות
            fig = px.histogram(df, x=target_col, marginal="rug", title=f"התפלגות של {target_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # מבחן שפירו-וילק
            data_clean = df[target_col].dropna()
            if len(data_clean) > 3:
                stat, p = stats.shapiro(data_clean)
                st.write(f"**מבחן Shapiro-Wilk:**")
                st.write(f"W={stat:.3f}, p={p:.4f}")
                if p > 0.05:
                    st.success("הנתונים מתפלגים נורמלית (p > 0.05)")
                else:
                    st.info("הנתונים אינם מתפלגים נורמלית (מומלץ לשקול מבחנים לא-פרמטריים)")
            else:
                st.error("אין מספיק נתונים לביצוע מבחן נורמליות.")

    st.divider()

    # 3. זיהוי ערכי קיצון (Outliers)
    st.markdown("### 📏 זיהוי ערכי קיצון (Outliers)")
    if numeric_cols:
        fig_box = px.box(df, y=numeric_cols, title="זיהוי חריגים במשתנים המספריים")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("נקודות מחוץ ל'שפמים' של התיבה נחשבות לערכי קיצון פוטנציאליים.")
