import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

from .utils import (
    ai_show, apa_table, significance_badge, store_result,
    label_eta, label_r, bonferroni_posthoc
)

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 המעבדה הסטטיסטית (StatsMonster Pro)")
    
    # הגדרת רשימות המשתנים
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    test = st.selectbox("בחר מבחן סטטיסטי:", [
        "📈 ניתוח שונות חד-כיווני (One-Way ANOVA)",
        "📊 ניתוח שונות דו-כיווני (Two-Way ANOVA)",
        "📉 רגרסיה לינארית (Linear Regression)",
        "🔗 מתאם פירסון (Pearson Correlation)"
    ])

    st.divider()
    col_setup, col_results = st.columns([1, 2], gap="large")

    # --- 1. One-Way ANOVA ---
    if "One-Way ANOVA" in test:
        with col_setup:
            # כאן החלפתי ל-all_cols כדי ש-Major יופיע
            gv = st.selectbox("משתנה קטגוריאלי (IV):", all_cols, key="anova_gv")
            dv = st.selectbox("משתנה תלוי (DV - חייב להיות מספרי):", numeric, key="anova_dv")
            show_post = st.checkbox("מבחני המשך (Bonferroni)")
            run_btn = st.button("▶️ הרץ ANOVA", use_container_width=True)

        with col_results:
            if run_btn:
                valid_df = df[[gv, dv]].dropna()
                groups = [g[dv].values for _, g in valid_df.groupby(gv)]
                f_stat, p = stats.f_oneway(*groups)
                
                n_valid = len(valid_df)
                df_b, df_w = len(groups)-1, n_valid - len(groups)
                eta2 = (f_stat * df_b) / (f_stat * df_b + df_w)

                st.subheader(f"📊 תוצאות: {significance_badge(p)}")
                apa_table([
                    ("F", f"{f_stat:.3f}"), 
                    ("df", f"({df_b}, {df_w})"), 
                    ("p-value", f"{p:.4f}"), 
                    ("η² (Effect Size)", f"{eta2:.3f}")
                ])
                
                if p < 0.05 and show_post:
                    st.write("**מבחני המשך:**")
                    st.dataframe(bonferroni_posthoc(valid_df, gv, dv))
                
                st.plotly_chart(px.box(valid_df, x=gv, y=dv, color=gv, points="all"), use_container_width=True)
                _handle_ai(f"One-Way ANOVA: F({df_b},{df_w})={f_stat:.3f}, p={p:.4f}, η²={eta2:.3f}", "anova")

    # --- 2. Two-Way ANOVA ---
    elif "Two-Way ANOVA" in test:
        with col_setup:
            f1 = st.selectbox("גורם א' (Major/Group):", all_cols, key="tw_f1")
            f2 = st.selectbox("גורם ב':", all_cols, key="tw_f2")
            dv = st.selectbox("משתנה תלוי (DV):", numeric, key="tw_dv")
            run_btn = st.button("▶️ הרץ ניתוח דו-כיווני", use_container_width=True)

        with col_results:
            if run_btn:
                valid_df = df[[f1, f2, dv]].dropna()
                try:
                    # המודל הסטטיסטי
                    formula = f'Q("{dv}") ~ C(Q("{f1}")) * C(Q("{f2}"))'
                    model = ols(formula, data=valid_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    # --- בניית טבלת SPSS Style ---
                    spss_table = anova_table.copy()
                    
                    # 1. חישוב Mean Square (חסר בברירת המחדל של פייתון)
                    spss_table['Mean Square'] = spss_table['sum_sq'] / spss_table['df']
                    
                    # 2. שינוי שמות עמודות לשמות המקובלים בתזה
                    spss_table.columns = ['Sum of Squares', 'df', 'F', 'Sig.', 'Mean Square']
                    
                    # 3. סידור מחדש של העמודות לפי הסדר של SPSS
                    spss_table = spss_table[['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.']]
                    
                    # 4. ניקוי שמות המשתנים (בלי ה-Q וה-C המעצבנים)
                    new_index = []
                    for name in spss_table.index:
                        clean_name = name.replace('C(Q("', '').replace('"))', '').replace(':', ' x ')
                        if clean_name == 'Residual': clean_name = 'Error (תוך קבוצתי)'
                        new_index.append(clean_name)
                    spss_table.index = new_index

                    st.subheader("📊 טבלת ANOVA (פורמט SPSS לתזה)")
                    
                    # תצוגה מעוצבת עם הדגשת מובהקות
                    st.table(spss_table.style.format({
                        'Sum of Squares': "{:.3f}",
                        'df': "{:.0f}",
                        'Mean Square': "{:.3f}",
                        'F': "{:.3f}",
                        'Sig.': "{:.4f}"
                    }))
                    
                    # הוספת הסבר קצר מתחת לטבלה
                    st.caption("הערה: מבחן ה-F חושב באמצעות Type II Sum of Squares כמקובל.")
                    
                    # המשך לגרף וצ'אט...
                    
                except Exception as e:
                    st.error(f"שגיאה בעיבוד הטבלה: {e}")

    # --- 3. Regression ---
    elif "Regression" in test:
        with col_setup:
            y_var = st.selectbox("משתנה תלוי (Y):", numeric)
            x_vars = st.multiselect("מנבאים (X):", numeric)
            run_btn = st.button("▶️ הרץ רגרסיה", use_container_width=True)

        with col_results:
            if run_btn and x_vars:
                valid_df = df[x_vars + [y_var]].dropna()
                X, y = valid_df[x_vars], valid_df[y_var]
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                st.subheader(f"📊 R² = {r2:.3f}")
                _handle_ai(f"Regression: R²={r2:.3f}, Predictors={x_vars}", "reg")

    # --- 4. Correlation ---
    elif "Correlation" in test:
        with col_setup:
            selected = st.multiselect("בחר משתנים:", numeric)
            run_btn = st.button("▶️ צור מטריצה", use_container_width=True)

        with col_results:
            if run_btn and len(selected) > 1:
                corr_matrix = df[selected].corr()
                fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
                _handle_ai(f"Correlation Matrix for: {selected}", "corr")

def _handle_ai(res_str, key):
    st.divider()
    st.subheader("🤖 צ'אט ניתוח ממצאים")
    
    # 1. אתחול היסטוריית השיחה אם היא לא קיימת
    chat_key = f"chat_history_{key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {"role": "assistant", "content": f"היי! ניתחתי את הנתונים. התוצאה היא: {res_str}. מה תרצה לדעת על הממצא הזה?"}
        ]

    # 2. תצוגת השיחה הקיימת
    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. שדה קלט לצ'אט
    if prompt := st.chat_input("שאל אותי משהו על הממצאים...", key=f"input_{key}"):
        # הוספת הודעת המשתמש להיסטוריה
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # יצירת תשובה מה-AI עם ההקשר של כל השיחה
        with st.chat_message("assistant"):
            try:
                # בניית ההקשר המלא ל-Gemini
                history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state[chat_key]])
                full_prompt = f"""
                אתה סטטיסטיקאי מומחה. זהו הקשר השיחה עד כה:
                {history_context}
                
                ענה על השאלה האחרונה של הסטודנט בצורה מקצועית, אקדמית ובעברית.
                """
                
                # שימוש בפונקציה מה-utils (וודא שהיא מחזירה את הטקסט)
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(full_prompt)
                
                answer = response.text
                st.markdown(answer)
                
                # שמירת תשובת ה-AI להיסטוריה
                st.session_state[chat_key].append({"role": "assistant", "content": answer})
                
                # שמירה לטאב של כתיבת התזה (אופציונלי)
                store_result(f"שאלה: {prompt} | תשובה: {answer}")
                
            except Exception as e:
                st.error(f"שגיאה בחיבור ל-AI: {e}")
