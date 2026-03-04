import streamlit as st
import pandas as pd
# ייבוא המנוע מהקובץ השני שלך
from ai_engine import render_ai_engine 

# הגדרת עמוד - חייב להיות הפקודה הראשונה של Streamlit
st.set_page_config(
    page_title="עוזר מחקר אקדמי AI",
    page_icon="🎓",
    layout="wide", # נותן יותר מרחב לטבלאות וגרפים
    initial_sidebar_state="expanded"
)

# עיצוב בסיסי ליישור לימין (RTL)
st.markdown("""
    <style>
    .reportview-container {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# --- תפריט צד (Sidebar) ---
with st.sidebar:
    st.title("📂 ניהול נתונים")
    st.write("העלה את קובץ המחקר שלך כאן:")
    
    uploaded_file = st.file_uploader(
        "בחר קובץ Excel או CSV", 
        type=["xlsx", "xls", "csv"],
        help="תומך בקבצי אקסל ו-CSV שהורדו מ-Qualtrics או SPSS"
    )
    
    if uploaded_file:
        st.success(f"✅ הקובץ '{uploaded_file.name}' נטען")
        
        if st.button("🗑️ נקה היסטוריית צ'אט"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()

# --- גוף האפליקציה המרכזי ---
if uploaded_file is not None:
    try:
        # טעינת הנתונים לפי סוג הקובץ
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # הפעלת הפונקציה המרכזית מקובץ ai_engine.py
        render_ai_engine(df)
        
    except Exception as e:
        st.error(f"⚠️ שגיאה בטעינת הקובץ: {e}")
        st.info("וודא שהקובץ אינו פתוח בתוכנה אחרת (כמו אקסל) ונסה שוב.")

else:
    # דף נחיתה מוצג כשאין קובץ טעון
    st.title("🎓 ברוכים הבאים לעוזר המחקר האקדמי")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("מה המערכת יודעת לעשות?")
        st.markdown("""
        * **ניתוח מגמות אישיות:** "הצג את ההתקדמות של תלמיד 10 במבחן המרחבי".
        * **השוואת קבוצות (ANOVA):** "האם יש הבדל בין בנים לבנות בציון הסופי?".
        * **סטטיסטיקה תיאורית:** הפקת טבלאות בסגנון SPSS בלחיצת כפתור.
        * **כתיבה אקדמית:** ניסוח הממצאים בפורמט APA 7 מוכן להעתקה.
        """)
    
    with col2:
        st.info("👈 **כדי להתחיל, העלה קובץ נתונים בתפריט הצד.**")
        # אופציונלי: הוספת תמונה או הסבר על מבנה הקובץ הרצוי
        

# --- כותרת תחתונה ---
st.sidebar.markdown("---")
st.sidebar.caption("פותח עבור חוקרים וסטודנטים | 2026")
