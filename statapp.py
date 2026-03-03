import streamlit as st
import pandas as pd
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
    # טעינת האקסל תוך התעלמות משורה 1 (שורת הציונים של ה-Q)
    # הערה: skiprows=[0] מדלג על השורה הראשונה של הנתונים
    df = pd.read_excel(uploaded_file, skiprows=[0])
    
    # תיקון שמות עמודות כפולים או ריקים (מונע DuplicateError)
    new_columns = []
    for i, col in enumerate(df.columns):
        clean_name = str(col).strip()
        if clean_name == "" or "Unnamed" in clean_name:
            new_columns.append(f"Column_{i}")
        else:
            new_columns.append(clean_name)
    df.columns = new_columns

    # ניסיון המרה אוטומטי למספרים
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success("הקובץ נטען בהצלחה (דילגנו על שורת הציונים)")

    # --- Variable View (כמו ב-SPSS) בתוך Sidebar ---
    st.sidebar.header("⚙️ Variable View")
    column_types = {}
    
    for col in df.columns:
        # זיהוי אוטומטי: אם זה מספר, נציע Scale
        is_num = pd.api.types.is_numeric_dtype(df[col])
        default_idx = 0 if is_num else 1
        
        column_types[col] = st.sidebar.selectbox(
            f"הגדר משתנה: {col}",
            options=["Scale (Numeric)", "Nominal (Text)"],
            index=default_idx,
            key=f"type_{col}"
        )

    # החלת סוגי הנתונים לפי בחירת המשתמש
    for col, v_type in column_types.items():
        if v_type == "Nominal (Text)":
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # סינון דאטה-פריים רק למספרים לצורך סטטיסטיקה תיאורית
    numeric_only_df = df.select_dtypes(include=['number'])
    
    # --- תצוגת הטאבים ---
    tab1, tab2, tab3 = st.tabs(["📋 נתונים", "📈 גרפים", "🤖 Gemini AI"])
    
    with tab1:
        st.subheader("סטטיסטיקה תיאורית (Scale בלבד)")
        if not numeric_only_df.empty:
            st.write(numeric_only_df.describe())
        else:
            st.warning("לא הוגדרו עמודות כ-Scale.")
        
        st.subheader("מבט על הנתונים")
        st.write(df.head())

    with tab2:
        visuals.render_visuals(df)
        
    with tab3:
        ai_engine.render_ai_analysis(df)
else:
    st.info("ממתין להעלאת קובץ אקסל...")
