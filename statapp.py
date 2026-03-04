import streamlit as st
import pandas as pd
import numpy as np

# --- פונקציית עזר לטעינה חכמה ---
def smart_load_excel(uploaded_file):
    # טעינה זמנית של תחילת הקובץ כדי להבין את המבנה
    preview = pd.read_excel(uploaded_file, nrows=5, header=None)
    
    # חישוב: איזו שורה נראית כמו כותרת (יותר טקסט ממספרים)
    header_row = 0
    for i in range(min(3, len(preview))):
        text_count = preview.iloc[i].apply(lambda x: isinstance(x, str)).sum()
        if text_count > (len(preview.columns) / 2):
            header_row = i
            break
    
    # טעינה סופית עם ה-Header הנכון
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, header=header_row)

if uploaded_file:
    # איפוס חסין במידה והקובץ הוחלף
    if uploaded_file.name != st.session_state.get('last_uploaded_file'):
        st.session_state.clear()
        st.session_state['last_uploaded_file'] = uploaded_file.name
        st.rerun()

    try:
        # 1. טעינה חכמה
        df = smart_load_excel(uploaded_file)
        
        # 2. ניקוי שמות עמודות
        df.columns = [str(c).strip().replace('.', '_') for c in df.columns]

        # 3. ניקוי עמודות ריקות לחלוטיn
        df = df.dropna(axis=1, how='all')

        # 4. המרה אוטומטית למספרים היכן שאפשר
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        st.success(f"✅ המנוע הסתגל לקובץ: '{uploaded_file.name}'")

        # --- Variable View אוטומטי לחלוטין ---
        with st.expander("⚙️ הגדרת משתנים אוטומטית (Variable View)"):
            var_data = []
            for col in df.columns:
                unique_vals = df[col].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                
                # כלל אצבע: אם יש פחות מ-10 ערכים ייחודיים, זה כנראה Nominal (קבוצה)
                is_nominal = not is_numeric or unique_vals < 10
                
                var_data.append({
                    "Variable": col,
                    "Type": "Scale (📏)" if is_numeric and not is_nominal else "Nominal (🧬)",
                    "Unique Values": unique_vals,
                    "Selected as Group?": is_nominal
                })
            
            settings_df = pd.DataFrame(var_data)
            edited_df = st.data_editor(
                settings_df,
                column_config={
                    "Variable": st.column_config.TextColumn("שם המשתנה", disabled=True),
                    "Type": st.column_config.TextColumn("סיווג אוטומטי", disabled=True),
                    "Unique Values": st.column_config.NumberColumn("ערכים ייחודיים", disabled=True),
                    "Selected as Group?": st.column_config.CheckboxColumn("בחר כמשתנה קבוצה (Major)")
                },
                hide_index=True,
                key=f"editor_{uploaded_file.name}"
            )

            # עדכון סופי של הנתונים לפני המעבדה
            for _, row in edited_df.iterrows():
                v_name = row["Variable"]
                if row["Selected as Group?"]:
                    df[v_name] = df[v_name].astype(str)
                else:
                    df[v_name] = pd.to_numeric(df[v_name], errors='coerce')

        # מעבר לטאבים
        tab_stats, tab_visuals, tab_ai = st.tabs(["🔬 מעבדה סטטיסטית", "📈 גרפים", "🤖 Gemini AI"])
        with tab_stats:
            render_stats_lab(df)

    except Exception as e:
        st.error(f"המנוע נתקל בקושי בקריאת הקובץ: {e}")
