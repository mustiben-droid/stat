import google.generativeai as genai
import streamlit as st

def ask_gemini(prompt, data_summary, last_stats):
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        full_prompt = f"""
        תפקיד: סטטיסטיקאי אקדמי בכיר.
        הקשר נתונים: {data_summary}
        תוצאות ניתוח: {last_stats}
        
        שאלה: {prompt}
        
        הנחיות: ענה בעברית אקדמית. השתמש בפורמט APA 7. הסבר מובהקות (p) וגודל אפקט (d).
        אם התוצאה לא מובהקת, הסבר את המשמעות המחקרית.
        """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"שגיאה: {e}"