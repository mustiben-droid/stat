import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import re

def render_ai_engine(df):
    # כותרת הטאב
    st.subheader("🤖 עוזר מחקר חכם")
    
    # 1. ניהול זיכרון ההודעות (חייב להיות מחוץ לכל תנאי)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. הצגת היסטוריית ההודעות בתוך קונטיינר כדי שלא יזוזו
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plot" in message:
                    st.plotly_chart(message["plot"], use_container_width=True)

    # 3. תיבת הקלט - ממוקמת מחוץ לקונטיינר כדי שתמיד תהיה בתחתית
    if prompt := st.chat_input("למשל: תראה לי שיפור של תלמיד 4"):
        
        # הצגת הודעת המשתמש מיד
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # תגובת ה-AI
        with chat_container:
            with st.chat_message("assistant"):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # בניית הקשר
                    context = f"Columns: {list(df.columns)}. User: {prompt}. Answer in Hebrew."
                    response = model.generate_content(context)
                    st.markdown(response.text)
                    
                    # לוגיקה לגרף (למשל תלמיד 4)
                    fig = None
                    id_match = re.search(r'(?:תלמיד|id|ID)\s*(\d+)', prompt)
                    
                    if id_match:
                        sid = int(id_match.group(1))
                        student_df = df[df['student_id'] == sid].copy()
                        if not student_df.empty:
                            if any(w in prompt for w in ["שיפור", "התקדמות", "זמן", "date"]):
                                student_df['date'] = pd.to_datetime(student_df['date'])
                                student_df = student_df.sort_values('date')
                                fig = px.line(student_df, x='date', y='score_proj', 
                                             title=f"שיפור בציון - תלמיד {sid}", markers=True)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # שמירה להיסטוריה
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.text, 
                        "plot": fig if fig else None
                    })
                    
                except Exception as e:
                    st.error(f"שגיאה: {e}")
