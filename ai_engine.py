import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import re

def render_ai_engine(df):
    st.markdown("### 🤖 עוזר מחקר AI פעיל")
    
    if df is None or df.empty:
        st.error("לא הועברו נתונים למנוע.")
        return

    # ניהול היסטוריה
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_placeholder = st.container(height=450)

    with chat_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plot" in message and message["plot"] is not None:
                    st.plotly_chart(message["plot"], use_container_width=True)
                if "data" in message and message["data"] is not None:
                    st.dataframe(message["data"])

    # תיבת הקלט
    if prompt := st.chat_input("למשל: תראה לי שיפור של תלמיד 4 ב-score_spatial"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_placeholder:
            with st.chat_message("assistant"):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # 1. שליחה ל-AI לקבלת תובנה מילולית
                    context = f"Columns: {list(df.columns)}. User wants: {prompt}. Answer in Hebrew."
                    response = model.generate_content(context)
                    st.markdown(response.text)
                    
                    # 2. לוגיקת חילוץ נתונים אקטיבית (החלק שחוסך לו "לבקש" נתונים)
                    fig = None
                    filtered_data = None
                    
                    # חילוץ ID מהטקסט (למשל "תלמיד 1" או "id 4")
                    id_match = re.search(r'(\d+)', prompt)
                    
                    if id_match:
                        sid = int(id_match.group(1))
                        # סינון הנתונים
                        filtered_data = df[df['student_id'] == sid].copy()
                        
                        if not filtered_data.empty:
                            st.success(f"חילוץ נתונים הצליח: נמצאו {len(filtered_data)} שורות עבור תלמיד {sid}")
                            
                            # אם יש תאריך, נסדר אותו לגרף
                            if 'date' in filtered_data.columns:
                                filtered_data['date'] = pd.to_datetime(filtered_data['date'], dayfirst=True, errors='coerce')
                                filtered_data = filtered_data.sort_values('date')
                                
                                # זיהוי איזה ציון להציג (מחפש מילים דומות בשאלה או לוקח spatial כברירת מחדל)
                                y_col = next((c for c in df.columns if c in prompt), 'score_spatial')
                                
                                if y_col in df.columns:
                                    fig = px.line(filtered_data, x='date', y=y_col, 
                                                 title=f"מגמת {y_col} - תלמיד {sid}", 
                                                 markers=True, template="plotly_white")
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"תלמיד {sid} לא נמצא בקובץ.")

                    # שמירת התוצאות להיסטוריה
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.text,
                        "plot": fig,
                        "data": filtered_data if filtered_data is not None else None
                    })
                    
                except Exception as e:
                    st.error(f"שגיאה: {e}")
