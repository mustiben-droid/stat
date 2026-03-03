import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# ייבוא המודולים מהתיקייה החדשה
from modules.stats_lab import render_stats_lab
# כאן תוכל לייבא את שאר המודולים שלך בעתיד:
# from modules.health_check import render_health_check

# הגדרות עמוד
st.set_page_config(page_title="Statistical Monster - Thesis Helper", layout="wide", page_icon="🧪")

# הגדרת ה-AI - תמיכה גם ב-Secrets וגם ב-Environment Variables
GEMINI_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    st.warning("⚠️ API Key לא נמצא. חלק מיכולות ה-AI לא יעבדו.")

st.title("🧪 Statistical Monster")
st.write("עוזר מחקר AI לסטודנטים לתזה - המעבדה הסטטיסטית שלך")

# העלאת קובץ בתפריט הצד
uploaded_file = st.sidebar.file_uploader("העלה קובץ אקסל (XLSX)", type=["xlsx"])

if uploaded_file:
    # 1. טעינת האקסל
    df = pd.read_excel(uploaded_file, header=1)
    
    # 2. ניקוי שמות העמודות
    new_columns = []
    for i, col in enumerate(df.columns):
        clean_name = str(col).strip()
        if clean_name == "" or "Unnamed" in clean_name:
            new_columns.append(f"Var_{i}")
        else:
            new_columns.append(clean_name)
    df.columns = new_columns

    # 3. המרה אוטומטית למספרים
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success(f"✅ הקובץ נטען! {len(df.columns)} משתנים זוהו.")

    # --- ה-Variable View החכם שלך ---
    with st.expander("⚙️ הגדרת סוגי משתנים (Variable View)"):
        var_settings = pd.DataFrame({
            "Variable": df.columns,
            "Is Nominal?": [not pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]
        })

        edited_settings = st.data_editor(
            var_settings,
            column_config={
                "Variable": st.column_config.TextColumn("שם המשתנה", disabled=True),
                "Is Nominal?": st.column_config.CheckboxColumn("Nominal (V) / Scale (ריק)")
            },
            hide_index=True,
            use_container_width=True
        )

        # עדכון ה-DF לפי הבחירה
        nominal_list = edited_settings[edited_settings["Is Nominal?"] == True]["Variable"].tolist()
        for col in df.columns:
            if col in nominal_list:
                df[col] = df[col].astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    st.divider()

    # --- ניהול הטאבים (כאן המעבדה החדשה נכנסת) ---
    tab_stats, tab_visuals, tab_ai = st.tabs(["🔬 מעבדה סטטיסטית", "📈 גרפים וניתוחים", "🤖 Gemini AI"])
    
    with tab_stats:
        # כאן אנחנו קוראים לקוד ה"מפלצתי" שבנינו ב-stats_lab.py
        render_stats_lab(df)
        
    with tab_visuals:
        st.subheader("גרפים ויזואליים")
        # אם יש לך קובץ visuals.py בתוך modules, קרא לו כך: 
        # modules.visuals.render_visuals(df)
        st.info("כאן יוצגו הגרפים הנוספים שלך.")
        
    with tab_ai:
        st.subheader("ייעוץ AI חופשי")
        # כאן יבוא מנוע ה-AI שלך
        st.info("כאן תוכל לשאול את Gemini שאלות על המחקר.")

else:
    st.info("👋 ברוך הבא! אנא העלה קובץ אקסל בתפריט הצד כדי להתחיל בניתוח.")
