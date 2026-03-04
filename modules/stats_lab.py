import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 Universal Statistic Analyzer")

    # --- שלב 1: ריענון וניקוי נתונים בכל פעם שהפונקציה רצה ---
    # הסרת עמודות ריקות לגמרי וניקוי רווחים
    df = df.dropna(axis=1, how='all')
    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
    
    # רשימות מעודכנות לקובץ הנוכחי
    all_vars = df.columns.tolist()
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()

    # הודעת סטטוס קטנה שתראה לך שהקובץ הוחלף בהצלחה
    st.toast(f"הקובץ נטען: {len(all_vars)} משתנים זוהו.", icon="✅")

    # --- שלב 2: זיהוי אוטומטי של ה-Major ---
    # אנחנו מחפשים עמודה שהיא לא "שאלה" (כלומר קצרה) או שהיא בסוף הקובץ
    # אם יש עמודה שקוראים לה "Major" או "Group" היא תיבחר, אם לא - העמודה הלפני אחרונה.
    potential_majors = [c for c in all_vars if 'major' in c.lower() or 'group' in c.lower() or 'class' in c.lower()]
    default_major = potential_majors[0] if potential_majors else all_vars[-5] if len(all_vars) > 5 else all_vars[-1]

    # --- שלב 3: ממשק המשתמש ---
    analysis_type = st.sidebar.radio("בחר ניתוח", [
        "🎯 Simple Main Effects", 
        "📈 ANOVA (Repeated)", 
        "📊 Descriptives",
        "🛡️ Reliability"
    ])

    st.divider()

    if analysis_type == "🎯 Simple Main Effects":
        st.subheader("ניתוח אפקטים פשוטים (SME)")
        
        col1, col2 = st.columns(2)
        with col1:
            # בוחר אוטומטית את המשתנה המספרי הראשון (בדרך כלל הציון)
            target = st.selectbox("בחר משתנה מטרה (ציון):", numeric_vars)
        with col2:
            # מציג את כל העמודות בקובץ כדי שלא תפספס את ה-Major
            group = st.selectbox("בחר משתנה קבוצה (Major):", all_vars, index=all_vars.index(default_major))
            
        if st.button("הרץ ניתוח"):
            try:
                # הכנה דינמית של הנתונים
                temp_df = df[[target, group]].dropna()
                temp_df[group] = temp_df[group].astype(str)
                
                # הרצה עם הגנה על שמות עמודות (Q)
                formula = f'Q("{target}") ~ C(Q("{group}"))'
                model = ols(formula, data=temp_df).fit()
                res = sm.stats.anova_lm(model, typ=3)
                
                st.write(f"### ANOVA: {group} בתוך המשתנה {target}")
                st.table(res.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))
                
                # עדכון זיכרון ה-AI לשיחה בהמשך
                st.session_state['global_context'] = f"SME analysis on {target} grouped by {group}."
            except Exception as e:
                st.error(f"שגיאה בהרצה: {e}")

    elif analysis_type == "📊 Descriptives":
        st.subheader("סטטיסטיקה תיאורית")
        selected_vars = st.multiselect("בחר משתנים:", numeric_vars)
        if selected_vars:
            st.table(df[selected_vars].describe().T.style.format("{:.2f}"))

    # (שאר המבחנים ימשיכו באותו מבנה דינמי)
