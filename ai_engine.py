import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import spearmanr

# ---------- פונקציות עזר סטטיסטיות (SPSS Style) ----------

def detect_structure(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = [
        c for c in df.columns
        if 'date' in str(c).lower() or 'time' in str(c).lower()
        or str(df[c].dtype).startswith('datetime')
    ]
    binary_cols = [c for c in num_cols if df[c].nunique() == 2]
    id_candidates = [c for c in df.columns if df[c].nunique() <= max(10, int(len(df) * 0.3))]
    return num_cols, cat_cols, date_cols, binary_cols, id_candidates

def spss_descriptives(df, cols):
    if not cols: return None
    desc = df[cols].describe().T
    desc = desc[['count', 'mean', 'std', 'min', 'max']]
    desc = desc.rename(columns={
        'count': 'N', 'mean': 'Mean', 'std': 'Std. Deviation',
        'min': 'Minimum', 'max': 'Maximum'
    }).round(3)
    return desc

def spss_anova(df, dv, group):
    try:
        temp = df[[dv, group]].dropna()
        if temp.empty: return None, None
        temp[group] = temp[group].astype(str)
        
        # הרצת המודל (Type III SS)
        model = ols(f'Q("{dv}") ~ C(Q("{group}"))', data=temp).fit()
        table = sm.stats.anova_lm(model, typ=3)
        
        # חישוב גודל אפקט Partial Eta Squared
        ss_effect = table.iloc[1]['sum_sq']
        ss_resid = table.iloc[2]['sum_sq']
        eta_sq = ss_effect / (ss_effect + ss_resid)
        
        # חישוב מדדי עזר עבור ה-AI (ממוצעים וסטיות תקן)
        desc_stats = temp.groupby(group)[dv].agg(['mean', 'std', 'count']).round(3)
        desc_stats.columns = ['Mean', 'SD', 'N']
        
        table = table.rename(columns={'sum_sq': 'SS', 'df': 'df', 'F': 'F', 'PR(>F)': 'Sig.'}).round(3)
        table['Partial Eta Sq'] = [np.nan, round(eta_sq, 3), np.nan, np.nan]
        
        return table, desc_stats
    except Exception as e:
        st.error(f"שגיאה בניתוח ANOVA: {e}")
        return None, None

# ---------- הפונקציה הראשית: render_ai_engine ----------

def render_ai_engine(df: pd.DataFrame):
    st.header("🧠 עוזר מחקר אקדמי חכם")

    if df is None or df.empty:
        st.error("לא נטענו נתונים.")
        return

    # 1. זיהוי מבנה הקובץ
    num_cols, cat_cols, date_cols, binary_cols, id_candidates = detect_structure(df)

    with st.expander("📊 סריקת מבנה הקובץ"):
        col1, col2 = st.columns(2)
        col1.write(f"**כמותיים:** {len(num_cols)}\n\n**קטגוריאליים:** {len(cat_cols)}")
        col2.write(f"**בינאריים:** {len(binary_cols)}\n\n**זמן:** {len(date_cols)}")

    # 2. בחירת מצב עבודה (Sidebar)
    mode = st.sidebar.radio(
        "בחר סוג ניתוח למסמך:",
        ["📝 פרשנות וניסוח אקדמי", "📊 סטטיסטיקה תיאורית", "⚖️ השוואת קבוצות (ANOVA)", "🔗 מתאמים (Spearman)"],
    )

    # 3. המלצות מחקר (פעם אחת לסשן)
    if "recommendations" not in st.session_state:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            rec_prompt = f"Analyze these columns: {list(df.columns)}. Suggest 3 research questions and statistical tests in Hebrew."
            st.session_state.recommendations = model.generate_content(rec_prompt).text
        except:
            st.session_state.recommendations = "לא ניתן לטעון המלצות כעת."

    st.info("💡 **כיווני מחקר מומלצים:**\n" + st.session_state.recommendations)

    # 4. ניהול הצ'אט והודעות
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None: st.table(msg["table"])
            if msg.get("desc") is not None: st.write("ממוצעים וסטיות תקן:"), st.table(msg["desc"])
            if msg.get("plot") is not None: st.plotly_chart(msg["plot"], use_container_width=True)

    # 5. קלט משתמש
    prompt = st.chat_input("שאל על הקובץ (למשל: 'השווה את Y לפי X' או 'תאר את המשתנים')")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # System Prompt אקדמי נוקשה (Ghostwriter)
            system_prompt = f"""
            You are a Senior Academic Researcher. Dataset columns: {list(df.columns)}.
            INSTRUCTIONS:
            1. Write a 'Results Section' in HEBREW.
            2. Follow APA 7th Edition: Report F, p, partial eta^2, M, and SD.
            3. Be formal and objective. 
            4. Do not explain Python code.
            """
            
            try:
                ai_model = genai.GenerativeModel('gemini-2.0-flash')
                response = ai_model.generate_content(f"{system_prompt}\n\nRequest: {prompt}")
                ai_text = response.text
            except:
                ai_text = "שגיאה בתקשורת עם ה-AI."

            st.markdown(ai_text)
            
            # לוגיקת ניתוח דינמית לפי מילות מפתח ובחירת המצב
            text_lower = prompt.lower()
            fig, table, desc = None, None, None

            # --- א. סטטיסטיקה תיאורית ---
            if mode == "📊 סטטיסטיקה תיאורית" or "תאר" in text_lower:
                table = spss_descriptives(df, num_cols)
                if table is not None: st.table(table)

            # --- ב. ANOVA ---
            elif mode == "⚖️ השוואת קבוצות (ANOVA)" or any(w in text_lower for w in ["השווה", "הבדל", "anova"]):
                # זיהוי משתנים חכם
                dv = next((c for c in num_cols if c.lower() in text_lower), num_cols[0] if num_cols else None)
                group = next((c for c in (binary_cols + cat_cols) if c.lower() in text_lower), (binary_cols + cat_cols)[0] if (binary_cols + cat_cols) else None)
                
                if dv and group:
                    table, desc = spss_anova(df, dv, group)
                    if table is not None:
                        st.table(table)
                        st.table(desc)
                        fig = px.box(df, x=group, y=dv, points="all", template="simple_white", title=f"התפלגות {dv} לפי {group}")
                        st.plotly_chart(fig)

            # --- ג. מתאמים ---
            elif mode == "🔗 מתאמים (Spearman)" or "קשר" in text_lower:
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr(method='spearman')
                    fig = px.imshow(corr, text_auto=True, title="מטריצת מתאמים", template="simple_white", color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig)

            # שמירה להיסטוריה
            st.session_state.messages.append({"role": "assistant", "content": ai_text, "table": table, "desc": desc, "plot": fig})

    # --- 6. כפתור ייצוא דוח מסכם ---
    if st.session_state.messages:
        if st.button("📄 הפק דו"ח מחקר מסכם (HTML/PDF)"):
            html = "<html><body dir='rtl' style='font-family: Arial;'><h1>דווח ממצאים אקדמי</h1>"
            for m in st.session_state.messages:
                html += f"<div style='border-bottom: 1px solid #ccc;'><p><b>{m['role']}:</b></p><div>{m['content']}</div>"
                if m.get('table') is not None: html += m['table'].to_html()
                html += "</div>"
            html += "</body></html>"
            st.download_button("הורד קובץ HTML", data=html, file_name="research_report.html", mime="text/html")
