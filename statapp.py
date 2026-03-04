import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# 1. Imports מהתיקייה החדשה
from modules.health_check   import render_health_check
from modules.test_wizard    import render_test_wizard
from modules.stats_lab      import render_stats_lab
from modules.thesis_writer  import render_thesis_writer
from modules.consultation   import render_consultation

# הגדרות עמוד
st.set_page_config(page_title="Statistical Monster - Thesis Helper", layout="wide", page_icon="🧪")

# --- מנגנון איפוס קבצים (תיקון קומט) ---
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

# הגדרת ה-AI
GEMINI_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

st.title("🧪 Statistical Monster")
st.write("עוזר מחקר AI לסטודנטים לתזה - המעבדה הסטטיסטית שלך")

# העלאת קובץ
uploaded_file = st.sidebar.file_uploader("העלה קובץ אקסל (XLSX)", type=["xlsx"])

if uploaded_file:
    # בדיקה האם הקובץ הוחלף
    if uploaded_file.name != st.session_state['last_uploaded_file']:
        # ניקוי ה-Session State פרט לשם הקובץ החדש
        st.session_state.clear()
        st.session_state['last_uploaded_file'] = uploaded_file.name
        st.rerun() # טעינה מחדש נקייה לחלוטין

    # 1. טעינת האקסל (הוספתי Try-Except למקרה של קובץ פגום)
    try:
        # header=1 אומר שהכותרות בשורה השנייה (מתאים לקבצי Google Forms מסוימים)
        df = pd.read_excel(uploaded_file, header=1)
        
        # 2. ניקוי שמות העמודות בצורה אגרסיבית
        new_columns = []
        for i, col in enumerate(df.columns):
            clean_name = str(col).strip()
            if clean_name == "" or "Unnamed" in clean_name or "nan" in clean_name.lower():
                new_columns.append(f"Var_{i}")
            else:
                new_columns.append(clean_name)
        df.columns = new_columns

        # 3. המרה ראשונית למספרים (בלי לאבד נתונים)
        df = df.apply(pd.to_numeric, errors='ignore')

        st.success(f"✅ הקובץ '{uploaded_file.name}' נטען! {len(df.columns)} משתנים זוהו.")

        # --- ה-Variable View החכם ---
        with st.expander("⚙️ הגדרת סוגי משתנים (Variable View)"):
            # יצירת מפתח ייחודי ל-editor שתלוי בשם הקובץ כדי שלא יזכור בחירות ישנות
            editor_key = f"editor_{uploaded_file.name}"
            
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
                use_container_width=True,
                key=editor_key
            )

            # עדכון ה-DF לפי הבחירה ב-Editor
            nominal_list = edited_settings[edited_settings["Is Nominal?"] == True]["Variable"].tolist()
            for col in df.columns:
                if col in nominal_list:
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        st.divider()

        # --- ניהול הטאבים ---
        tab_stats, tab_visuals, tab_ai = st.tabs(["🔬 מעבדה סטטיסטית", "📈 גרפים", "🤖 Gemini AI"])
        
        with tab_stats:
            # שליחת ה-DF המעודכן למעבדה
            render_stats_lab(df)
            
        with tab_visuals:
            st.info("כאן יוצגו הגרפים הנוספים שלך.")
            
        with tab_ai:
            st.subheader("ייעוץ AI חופשי")
            # העברת שמות העמודות ל-AI כדי שיכיר את המייג'ור והמשתנים
            if st.button("נתח את מבנה הקובץ שלי (AI)"):
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"הנה רשימת המשתנים שלי: {list(df.columns)}. מי מהם נראה לך כמו המייג'ור או משתנה בלתי תלוי? ענה בקצרה בעברית."
                response = model.generate_content(prompt)
                st.write(response.text)

    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")

else:
    st.info("👋 ברוך הבא! אנא העלה קובץ אקסל בתפריט הצד כדי להתחיל בניתוח.")
    st.session_state['last_uploaded_file'] = None
