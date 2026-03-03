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
    
    # ניסיון המרה אוטומטי ראשוני
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success("הקובץ נטען בהצלחה!")

    # --- ה-Variable View שלכם (כמו ב-SPSS) ---
    st.sidebar.header("⚙️ Variable View (הגדרת משתנים)")
    
    column_types = {}
    for col in df.columns:
        # זיהוי אוטומטי ראשוני להמלצה
        default_type = "Scale (Numeric)" if pd.api.types.is_numeric_dtype(df[col]) else "Nominal (Text)"
        
        # בחירת סוג המשתנה לכל עמודה
        column_types[col] = st.sidebar.selectbox(
            f"סוג המשתנה עבור {col}:",
            options=["Scale (Numeric)", "Nominal (Text)"],
            index=0 if default_type == "Scale (Numeric)" else 1,
            key=f"type_{col}"
        )

    # החלת השינויים על ה-DataFrame
    for col, v_type in column_types.items():
        if v_type == "Nominal (Text)":
            df[col] = df[col].astype(str)
        else:
            # ניסיון להפוך חזרה למספר אם המשתמש התחרט
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # יצירת DataFrame מסונן לחישובים (רק מה שהוגדר כ-Scale)
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


