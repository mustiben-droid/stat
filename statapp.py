import streamlit as st
import pandas as pd
import google.generativeai as genai
from ai_engine import render_ai_engine

# 1. הגדרות עמוד (חייב להיות ראשון)
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. הגדרת מפתח API אוטומטית מתוך Secrets
if "api_key" not in st.session_state:
    try:
        # שליפת המפתח מה-Secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        st.session_state.api_key = api_key
    except Exception:
        st.error("❌ מפתח API לא נמצא! וודא שהגדרת GOOGLE_API_KEY ב-Secrets.")
        st.stop()

# 3. עיצוב RTL ויישור לימין
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Assistant', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stSidebar {
        direction: rtl;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. תפריט צד (Sidebar) - נקי ופונקציונלי
with st.sidebar:
    st.title("📂 ניהול מחקר")
    st.info("המערכת מחוברת ומוכנה לניתוח")
    
    uploaded_file = st.file_uploader("העלה קובץ נתונים (Excel/CSV)", type=["xlsx", "csv"])
    
    if uploaded_file:
        st.success(f"✅ נטען: {uploaded_file.name}")
        if st.button("🗑️ נקה היסטוריית מחקר"):
            st.session_state.messages = []
            st.rerun()

# 5. לוגיקה מרכזית
def main():
    if uploaded_file is not None:
        try:
            # קריאת הקובץ לפי סוג
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if df.empty:
                st.error("הקובץ שהועלה ריק.")
                return

            # הפעלת ה-AI Agent מתוך ai_engine.py
            render_ai_engine(df)
            
        except Exception as e:
            st.error(f"שגיאה בתהליך טעינת הנתונים: {e}")
    else:
        # דף נחיתה הסברי
        st.title("🎓 עוזר מחקר אקדמי חכם")
        st.subheader("כלי לניתוח סטטיסטי, ויזואליזציה וכתיבת ממצאים")
        
        st.markdown("""
        ### ברוך הבא למערכת הניתוח
        המערכת מחוברת אוטומטית ל-AI ומוכנה לעבודה. 
        
        **איך מתחילים?**
        1. **מעלים קובץ נתונים** בתפריט הצד (Excel או CSV).
        2. **שואלים שאלות:** המערכת תזהה אם ברצונך לראות מגמות של תלמידים, לבצע השוואות (ANOVA) או לקבל הסברים סטטיסטיים.
        
        המערכת תפיק עבורך גרפים אינטראקטיביים וניסוחים מוכנים לפרק הממצאים בפורמט APA 7.
        """)

if __name__ == "__main__":
    main()
