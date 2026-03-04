import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import re
import json
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- 1. פונקציית עזר לציור גרף אישי ---
def plot_student_trend(df, student_id, score_col):
    student_col = next((c for c in df.columns if 'id' in str(c).lower()), None)
    date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'time' in str(c).lower()), None)
    
    if not student_col or not date_col or not score_col:
        return None

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], dayfirst=True, errors="coerce")
    
    # סינון תלמיד ומיון בזמן
    student_data = temp[temp[student_col] == student_id].dropna(subset=[score_col, date_col])
    if student_data.empty:
        return None
        
    student_data = student_data.sort_values(date_col)
    
    fig = px.line(
        student_data, x=date_col, y=score_col, 
        markers=True, title=f"מגמת {score_col} - תלמיד {student_id}",
        template="simple_white"
    )
    return fig

# --- 2. המנוע הראשי עם ההגנות של פרפלקסיטי ---
def render_ai_engine(df):
    st.header("🎓 מנוע מחקר אישי וקבוצתי")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # תצוגת היסטוריה
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("plot") is not None:
                st.plotly_chart(msg["plot"], use_container_width=True)
            if msg.get("table") is not None:
                st.table(msg["table"])

    if prompt := st.chat_input("שאל על מגמה אישית או ניתוח קבוצתי..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # --- שלב החילוץ (עם הגנות Perplexity) ---
            extract_prompt = f"""
            Columns: {list(df.columns)}. Request: "{prompt}".
            Return ONLY JSON: {{"student_id": number or null, "target_col": "name" or null, "type": "trend" or "group"}}
            """
            
            student_id, target_col, analysis_type = None, None, "group"
            try:
                raw_json = model.generate_content(extract_prompt).text
                match = re.search(r'\{.*\}', raw_json, re.DOTALL)
                if match:
                    params = json.loads(match.group())
                    student_id = params.get("student_id")
                    target_col = params.get("target_col")
                    analysis_type = params.get("type", "group")
            except Exception:
                pass # נשאר עם ברירות מחדל

            fig, table = None, None
            
            # --- שלב הביצוע ---
            # 1. ניתוח מגמה אישית
            if analysis_type == "trend" and student_id is not None:
                # הגנה: אם המודל לא מצא עמודה מדויקת, ניקח את המספרית הראשונה
                if target_col not in df.columns:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    target_col = num_cols[0] if num_cols else None
                
                if target_col:
                    fig = plot_student_trend(df, student_id, target_col)
                
                if fig is not None:
                    st.plotly_chart(fig)
                    ai_text = f"מציג את גרף המגמה של תלמיד {student_id} עבור המשתנה {target_col}. הניתוח מבוסס על סדר כרונולוגי של הנתונים."
                else:
                    ai_text = f"זיהיתי שביקשת מגמה עבור תלמיד {student_id}, אך לא מצאתי נתונים מספיקים בקובץ לביצוע הניתוח."

            # 2. ניתוח קבוצתי (ANOVA)
            elif any(w in prompt.lower() for w in ["השוואה", "הבדל", "anova"]):
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = [c for c in df.columns if df[c].nunique() < 10 and c not in num_cols]
                
                if num_cols and cat_cols:
                    dv, iv = num_cols[0], cat_cols[0]
                    model_ols = ols(f'Q("{dv}") ~ C(Q("{iv}"))', data=df).fit()
                    table = sm.stats.anova_lm(model_ols, typ=3).round(3)
                    st.table(table)
                    ai_text = f"בוצע ניתוח ANOVA לבחינת הבדלים ב-{dv} לפי קבוצות ב-{iv}. התוצאות מופיעות בטבלה."
                else:
                    ai_text = "לא נמצאו משתנים מתאימים (כמותי וקטגוריאלי) לביצוע ANOVA."

            # 3. מענה חופשי
            else:
                ai_text = model.generate_content(f"ענה בעברית על: {prompt}. התבסס על העמודות: {list(df.columns)}").text

            st.markdown(ai_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_text, "plot": fig, "table": table})
