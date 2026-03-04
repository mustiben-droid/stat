import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import re

def render_ai_engine(df):
    st.markdown("### 🤖 מנוע ניתוח נתונים אוניברסלי")
    
    if df is None or df.empty:
        st.error("לא הועלו נתונים לניתוח.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # תצוגת הצ'אט
    chat_placeholder = st.container(height=450)
    with chat_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plot" in message and message["plot"] is not None:
                    st.plotly_chart(message["plot"], use_container_width=True)

    if prompt := st.chat_input("שאל כל שאלה על הקובץ שהעלית..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_placeholder:
            with st.chat_message("assistant"):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # --- זה הפרומפט האוניברסלי שמתאים לכל קובץ ---
                    columns_list = ", ".join(df.columns.tolist())
                    system_instructions = f"""
                    You are a professional data analyst. 
                    Current dataset columns: {columns_list}.
                    
                    Your instructions:
                    1. The data is already loaded in the system. Never tell the user to load it.
                    2. If the user asks for a specific ID, filter the data by the ID column (e.g., student_id, user_id, etc.).
                    3. If the user asks for 'improvement' or 'trends', use the date/time column for the X-axis.
                    4. Analyze the specific columns mentioned in the user's prompt.
                    5. Answer in HEBREW only, be direct and scientific.
                    6. Do NOT provide code or instructions. Simply confirm you are performing the analysis.
                    """

                    response = model.generate_content(f"{system_instructions}\n\nUser Question: {prompt}")
                    st.markdown(response.text)
                    
                    # --- מנגנון ביצוע גרפים אוטומטי ---
                    fig = None
                    
                    # זיהוי מספר (ID) בשאלה
                    id_match = re.search(r'(\d+)', prompt)
                    
                    # חיפוש עמודת מזהה (ID) כלשהי
                    id_col = next((c for c in df.columns if 'id' in c.lower() or 'key' in c.lower()), df.columns[0])
                    # חיפוש עמודת זמן כלשהי
                    date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'תאריך' in c), None)
                    # חיפוש עמודת ערך (Score/Value) מהשאלה
                    val_col = next((c for c in df.columns if c.lower() in prompt.lower()), None)

                    if id_match:
                        target_id = int(id_match.group(1))
                        # סינון לפי ה-ID שנמצא
                        filtered_df = df[df[id_col].astype(str) == str(target_id)].copy()
                        
                        if not filtered_df.empty and date_col and val_col:
                            # המרת תאריך לגרף תקין
                            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], dayfirst=True, errors='coerce')
                            filtered_df = filtered_df.sort_values(date_col)
                            
                            fig = px.line(filtered_df, x=date_col, y=val_col, 
                                         title=f"מגמת {val_col} עבור {id_col} {target_id}",
                                         markers=True, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                        elif not filtered_df.empty and not date_col:
                            st.write("הנתונים נמצאו, אך לא זוהתה עמודת תאריך להצגת מגמה.")
                            st.dataframe(filtered_df.head(10))

                    st.session_state.messages.append({"role": "assistant", "content": response.text, "plot": fig})
                    
                except Exception as e:
                    st.error(f"שגיאה בניתוח הנתונים: {e}")
