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

# 2. הגדרת מפתח API (מומלץ להשתמש ב-Secrets של Streamlit או להזין כאן)
# אם יש לך מפתח קבוע, שים אותו במקום ה-YOUR_API_KEY
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# 3. יישור לימין (RTL) ועיצוב אקדמי בעזרת CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Assistant', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f0f2f6;
    }
    .stTextInput>div>div>input {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. תפריט צד (Sidebar)
with st.sidebar:
    st.title("⚙️ הגדרות מערכת")
    
    # הזנת מפתח API במידה ולא מוגדר ב-Secrets
    user_key = st.text_input("הזן Google API Key:", type="password")
    if user_key:
        st.session_state.api_key = user_key
        genai.configure(api_key=user_key)
    
    st.divider()
    
    st.header("📂 טעינת נתונים")
    uploaded_file = st.file_uploader("בחר קובץ אקסל או CSV", type=["xlsx", "csv"])
    
    if uploaded_file:
        st.success(f"קובץ נטען: {uploaded_file.name}")
        if st.button("🗑️ נקה היסטוריית מחקר"):
            st.session_state.messages = []
            st.rerun()

# 5. לוגיקה מרכזית
def main():
    if not st.session_state.api_key:
        st.warning("⚠️ נא להזין מפתח API בתפריט הצד כדי להפעיל את יכולות ה-AI.")
        st.info("ניתן להנפיק מפתח בחינם בכתובת: https://aistudio.google.com/app/apikey")
        return

    if uploaded_file is not None:
        try:
            # קריאת הקובץ
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # בדיקה שהקובץ לא ריק
            if df.empty:
                st.error("הקובץ שהועלה ריק.")
                return

            # הפעלת המנוע מתוך ai_engine.py
            render_ai_engine(df)
            
        except Exception as e:
            st.error(f"שגיאה בתהליך טעינת הנתונים: {e}")
    else:
        # דף נחיתה
        st.title("🎓 עוזר מחקר אקדמי חכם")
        st.subheader("כלי לניתוח סטטיסטי, ויזואליזציה וכתיבת ממצאים")
        
        st.markdown("""
        ### איך זה עובד?
        1. **מזינים מפתח API** (בצד ימין למעלה).
        2. **מעלים קובץ נתונים** (Excel או CSV).
        3. **מתחילים לחקור:** ניתן לבקש מה-AI לנתח מגמות של תלמידים ספציפיים, לבצע מבחני ANOVA או להפיק סטטיסטיקה תיאורית.
        
        המערכת תייצר עבורך גרפים אינטראקטיביים וניסוחים מוכנים לפרק הממצאים בתזה/עבודה שלך.
        """)
        

if __name__ == "__main__":
    main()
