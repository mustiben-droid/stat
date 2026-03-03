import streamlit as st
import google.generativeai as genai

def render_ai_analysis(df):
    st.subheader("🤖 ניתוח חכם באמצעות Gemini AI")
    
    # הסבר קצר למשתמש
    st.info("ה-AI ינתח את הסטטיסטיקה התיאורית של הנתונים שלך כדי להפיק תובנות.")

    user_question = st.text_input("מה תרצה לדעת על הנתונים שלך? (למשל: 'מהן המסקנות העיקריות מהנתונים?')")
    
    if st.button("שאל את ה-AI"):
        if user_question:
            try:
                # יצירת המודל
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # הכנת הנתונים למודל (שליחת התיאור הסטטיסטי של הטבלה)
                stats_summary = df.describe().to_string()
                
                prompt = f"""
                אתה עוזר מחקר סטטיסטי מומחה. 
                להלן נתונים סטטיסטיים של המחקר שלי:
                {stats_summary}
                
                השאלה של המשתמש: {user_question}
                
                אנא ענה בעברית בצורה מקצועית, ברורה ומבוססת על הנתונים.
                """
                
                with st.spinner("ה-AI חושב..."):
                    response = model.generate_content(prompt)
                    st.markdown("---")
                    st.write("### תשובת ה-AI:")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"אירעה שגיאה בתקשורת עם ה-AI: {e}")
        else:
            st.warning("בבקשה כתוב שאלה לפני שאלחץ.")
