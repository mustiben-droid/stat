import streamlit as st
import pandas as pd
import google.generativeai as genai
from ai_engine import render_ai_engine

# 1. הגדרות עמוד
st.set_page_config(
    page_title="עוזר מחקר סטטיסטי",
    page_icon="🎓",
    layout="wide"
)

# 2. משיכת ה-API Key מה-Secrets (כניסה אוטומטית)
try:
    # וודא שב-Secrets שלך המפתח מוגדר תחת השם GOOGLE_API_KEY
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    st.session_state.api_key = api_key
except Exception as e:
    st.error("לא נמצא מפתח API ב-Secrets. וודא שהגדרת GOOGLE_API_KEY.")
    st.stop()

# 3. עיצוב RTL
st.markdown("""<style>body { direction: rtl; text-align: right; }</style>""", unsafe_allow_html=True)

# 4. תפריט צד
st.sidebar.title("📂 ניהול נתונים")
uploaded_file = st.sidebar.file_uploader("העלה קובץ (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if st.sidebar.button("🗑️ נקה היסטוריית צ'אט"):
        st.session_state.messages = []
        st.rerun()

# 5. לוגיקה מרכזית
if uploaded_file is not None:
    try:
        # טעינת הקובץ
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # הרצת המנוע
        render_ai_engine(df)
        
    except Exception as e:
        st.error(f"שגיאה בטעינת הקובץ: {e}")
else:
    st.title("🎓 עוזר מחקר אקדמי חכם")
    st.info("המערכת מחוברת בבטחה. העלה קובץ בתפריט הצד כדי להתחיל.")
