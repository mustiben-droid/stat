import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import re

def render_ai_engine(df):
    st.subheader("🤖 מנוע ניתוח נתונים מבוסס פקודות")
    st.info("ניתן לכתוב פקודות חופשיות, למשל: 'תראה לי גרף שיפור של תלמיד 4 בציון score_proj'")

    # ניהול זיכרון הצ'אט
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # הצגת היסטוריה
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)

    # קלט משתמש
    if prompt := st.chat_input("איזו בדיקה להריץ עבורך?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # יצירת הקשר ל-AI על מבנה הנתונים
                context = f"""
                You are a data scientist. User Dataframe columns: {list(df.columns)}.
                Sample values: {df.head(2).to_dict()}
                User Request: {prompt}
                
                Guidelines:
                1. If asked about a specific student ID, look for 'student_id'.
                2. If asked about progress or time, use 'date'.
                3. If you identify a specific analysis, explain what you found in Hebrew.
                """
                
                response = model.generate_content(context)
                st.markdown(response.text)
                
                # --- מנגנון ביצוע פקודות (Execution Engine) ---
                fig = None
                
                # חיפוש מזהה תלמיד (למשל "תלמיד 1" או "id 4")
                id_match = re.search(r'(?:תלמיד|סטודנט|id|ID)\s*(\d+)', prompt)
                
                if id_match:
                    target_id = int(id_match.group(1))
                    # סינון הנתונים לתלמיד הספציפי
                    student_df = df[df['student_id'] == target_id].copy()
                    
                    if not student_df.empty:
                        # בדיקה אם ביקשו שיפור/התקדמות/זמן
                        if any(w in prompt for w in ["שיפור", "התקדמות", "זמן", "date", "תאריך"]):
                            if 'date' in df.columns:
                                # המרה לתאריך ומיון
                                student_df['date'] = pd.to_datetime(student_df['date'])
                                student_df = student_df.sort_values('date')
                                
                                # בחירת עמודת הציון (לפי מה שהמשתמש כתב או ברירת מחדל)
                                y_col = next((c for c in df.columns if c in prompt), 'score_proj')
                                
                                if y_col in df.columns:
                                    fig = px.line(student_df, x='date', y=y_col, 
                                                 title=f"גרף התקדמות: תלמיד {target_id} (משתנה {y_col})",
                                                 markers=True, template="plotly_white")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.success(f"ניתוח הושלם עבור תלמיד {target_id}.")
                    else:
                        st.warning(f"לא מצאתי נתונים עבור תלמיד שמספרו {target_id}")

                # שמירה להיסטוריה
                msg_entry = {"role": "assistant", "content": response.text}
                if fig: msg_entry["plot"] = fig
                st.session_state.messages.append(msg_entry)

            except Exception as e:
                st.error(f"שגיאה במנוע ה-AI: {e}")
