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
st.write("העלה קובץ אקסל, הגדר משתנים וקבל תובנות מהירות")

# העלאת קובץ בתפריט הצד
uploaded_file = st.sidebar.file_uploader("העלה קובץ אקסל (XLSX)", type=["xlsx"])

if uploaded_file:
    # 1. טעינת האקסל - דילוג על שורה 1 (שורת הציונים של ה-Q)
    # הערה: אם שורת הציונים היא השורה הראשונה ממש (הכותרות), השתמש ב-header=1.
    # כאן אנחנו מדלגים על שורת הנתונים הראשונה שאחרי הכותרות:
    df = pd.read_excel(uploaded_file, skiprows=[1])
    
    # 2. תיקון שמות עמודות כפולים או ריקים
    new_columns = []
    for i, col in enumerate(df.columns):
        clean_name = str(col).strip()
        if clean_name == "" or "Unnamed" in clean_name:
            new_columns.append(f"Column_{i}")
        else:
            new_columns.append(clean_name)
    df.columns = new_columns

    # 3. ניסיון המרה אוטומטי ראשוני למספרים
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success("הקובץ נטען בהצלחה!")

    # --- ה-Variable View החדש והנוח ---
    st.markdown("### ⚙️ Variable View (הגדרת סוגי משתנים)")
    st.info("סמן ב-V את העמודות שהן קטגוריות (Nominal). עמודות ללא V ייחשבו כמספריות (Scale).")

    # יצירת טבלת הגדרות מבוססת על סוג הנתונים הנוכחי
    var_settings = pd.DataFrame({
        "Variable": df.columns,
        "Is Nominal?": [not pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]
    })

    # הצגת טבלה אינטראקטיבית לעריכה מהירה
    edited_settings = st.data_editor(
        var_settings,
        column_config={
            "Variable": st.column_config.TextColumn("שם המשתנה", disabled=True),
            "Is Nominal?": st.column_config.CheckboxColumn("Nominal (V) / Scale (ריק)")
        },
        hide_index=True,
        use_container_width=True
    )

    # 4. החלת השינויים על ה-DataFrame לפי הבחירה בטבלה
    nominal_list = edited_settings[edited_settings["Is Nominal?"] == True]["Variable"].tolist()
    
    for col in df.columns:
        if col in nominal_list:
            df[col] = df[col].astype(str)
        else:
            # הפיכה למספר, וערכים לא חוקיים יהפכו ל-NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # יצירת DataFrame מסונן רק למספרים לצורך הסטטיסטיקה התיאורית
    numeric_only_df = df.select_dtypes(include=['number'])
    
    st.divider()

    # --- תצוגת הטאבים ---
    tab1, tab2, tab3 = st.tabs(["📋 נתונים וסטטיסטיקה", "📈 גרפים וניתוחים", "🤖 Gemini AI"])
    
    with tab1:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("סטטיסטיקה תיאורית")
            if not numeric_only_df.empty:
                st.write(numeric_only_df.describe().T) # T הופך את הטבלה לנוחות קריאה
            else:
                st.warning("לא נבחרו עמודות מספריות (Scale).")
        
        with col_b:
            st.subheader("מבט על בסיס הנתונים")
            st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        # שליחת ה-df המעובד לפונקציית הגרפים
        visuals.render_visuals(df)
        
    with tab3:
        # שליחת ה-df לניתוח ה-AI
        ai_engine.render_ai_analysis(df)

else:
    st.info("אנא העלה קובץ אקסל דרך תפריט הצד כדי להתחיל.")
