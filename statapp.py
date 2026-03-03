import streamlit as st
import pandas as pd
import logic
import visuals
import ai_engine
import google.generativeai as genai

# הגדרות עמוד
st.set_page_config(page_title="StatsBot - Thesis Helper", layout="wide")

# הגדרת ה-AI מתוך ה-Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"שגיאה בהגדרת ה-API: {e}")

st.title("📊 מנוע ניתוח סטטיסטי לתזה")
st.write("העלה קובץ אקסל וקבל ניתוחים ותובנות AI")

uploaded_file = st.sidebar.file_uploader("העלה קובץ אקסל (XLSX)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # ניסיון המרה אוטומטי ראשוני למספרים
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success("הקובץ נטען בהצלחה!")

    # --- בורר עמודות נומינליות בתפריט הצד ---
    st.sidebar.header("⚙️ הגדרות משתנים")
    all_cols = df.columns.tolist()
    
    nominal_cols = st.sidebar.multiselect(
        "בחר עמודות שהן קטגוריות (נומינלי):", 
        options=all_cols,
        help="עמודות אלו יהפכו לטקסט ולא ייכללו בחישובים מספריים"
    )

    # המרת העמודות הנבחרות לטקסט
    for col in nominal_cols:
        df[col] = df[col].astype(str)

    # יצירת רשימה מעודכנת של עמודות מספריות (לצורך הסטטיסטיקה התיאורית)
    numeric_only_df = df.select_dtypes(include=['number'])
    
    # --- תצוגת הטאבים ---
    tab1, tab2, tab3 = st.tabs(["📋 נתונים", "📈 גרפים", "🤖 Gemini AI"])
    
    with tab1:
        st.subheader("סטטיסטיקה תיאורית (מספרי בלבד)")
        if not numeric_only_df.empty:
            st.write(numeric_only_df.describe())
        else:
            st.warning("לא נמצאו עמודות מספריות להצגת סטטיסטיקה.")
        
        st.subheader("מבט על הטבלה")
        st.write(df.head())

    with tab2:
        # הפונקציה ב-visuals כבר תדע לסנן רק את המספריות שנשארו
        visuals.render_visuals(df)
        
    with tab3:
        ai_engine.render_ai_analysis(df)
else:
    st.info("ממתין להעלאת קובץ...")

