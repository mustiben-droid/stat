import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import re

def render_ai_engine(df):
    st.markdown("### 🤖 מנוע ניתוח נתונים בזמן אמת")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_placeholder = st.container(height=450)

    with chat_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plot" in message and message["plot"] is not None:
                    st.plotly_chart(message["plot"], use_container_width=True)

    if prompt := st.chat_input("למשל: תראה לי שיפור של תלמיד 4"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_placeholder:
            with st.chat_message("assistant"):
                # --- שלב 1: חילוץ ID וביצוע פילטר (לפני ה-AI) ---
                fig = None
                id_match = re.search(r'(\d+)', prompt)
                
                if id_match:
                    sid = int(id_match.group(1))
                    # סינון אמיתי מהקובץ שלך
                    student_data = df[df['student_id'].astype(str) == str(sid)].copy()
                    
                    if not student_data.empty:
                        # זיהוי עמודת ציון מהשאלה או ברירת מחדל
                        y_col = next((c for c in df.columns if c in prompt), 'score_spatial')
                        
                        # סידור ציר הזמן
                        if 'date' in student_data.columns:
                            student_data['date'] = pd.to_datetime(student_data['date'], dayfirst=True, errors='coerce')
                            student_data = student_data.sort_values('date')
                            
                            # יצירת הגרף האמיתי מהנתונים
                            fig = px.line(student_data, x='date', y=y_col, 
                                         title=f"נתונים אמיתיים: מגמת {y_col} - תלמיד {sid}", 
                                         markers=True, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                            analysis_text = f"מצאתי את הנתונים עבור תלמיד {sid}. הנה גרף המגמה המבוסס על {len(student_data)} רשומות מהקובץ שלך."
                    else:
                        analysis_text = f"חיפשתי את תלמיד {sid} בקובץ, אך לא נמצאו נתונים תואמים."
                else:
                    # אם לא צוין ID, ניתן ל-AI לענות תשובה כללית
                    try:
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        context = f"Columns: {list(df.columns)}. Answer in Hebrew."
                        response = model.generate_content(f"{context}\nQuestion: {prompt}")
                        analysis_text = response.text
                    except:
                        analysis_text = "לא הצלחתי לנתח את הבקשה. נסה לציין מספר תלמיד (למשל: תלמיד 4)."

                st.markdown(analysis_text)
                st.session_state.messages.append({"role": "assistant", "content": analysis_text, "plot": fig})
