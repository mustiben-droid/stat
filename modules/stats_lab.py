import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import google.generativeai as genai
import statsmodels.api as sm
from statsmodels.formula.api import ols

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 מעבדת סטטיסטיקה מתקדמת - Full Report Mode")
    
    df.columns = df.columns.str.strip()
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # תפריט ה-Analyze המקצועי
    test = st.selectbox("בחר ניתוח (Analyze):", [
        "📊 Repeated Measures ANOVA (Pre vs Post Analysis)",
        "🧪 מבחן t למדגמים בלתי תלויים",
        "🔲 מבחן Chi-Square (קשר בין קטגוריות)",
        "🛡️ בדיקת מהימנות (Cronbach's Alpha)",
        "📈 מטריצת מתאמים וסטטיסטיקה תיאורית"
    ])

    st.divider()
    col_setup, col_results = st.columns([1, 2.5], gap="large")
    analysis_summary = ""

    # --- 1. REPEATED MEASURES ANOVA - הגרסה המורחבת ---
    if "Repeated Measures" in test:
        with col_setup:
            st.subheader("⚙️ נתוני ANOVA")
            pre = st.selectbox("זמן 1 (Pre):", numeric, key="rm_pre")
            post = st.selectbox("זמן 2 (Post):", numeric, key="rm_post")
            between = st.selectbox("גורם בין-נבדקי (Major):", all_cols, key="rm_btw")
            st.info("הניתוח כולל: ממוצעים, מובהקות, וגודל אפקט (Eta Squared).")
            run = st.button("🚀 Run Analysis", use_container_width=True)

        with col_results:
            if run:
                try:
                    # עיבוד נתונים לפורמט ארוך
                    temp_df = df[[pre, post, between]].dropna().copy()
                    temp_df['ID'] = range(len(temp_df))
                    long_df = pd.melt(temp_df, id_vars=['ID', between], value_vars=[pre, post], 
                                     var_name='Time', value_name='Score')
                    
                    # 1. טבלת ממוצעים (Descriptive Statistics)
                    st.write("### Descriptive Statistics")
                    desc = temp_df.groupby(between)[[pre, post]].agg(['mean', 'std', 'count'])
                    st.table(desc.style.format("{:.2f}"))

                    # 2. הרצת המודל הסטטיסטי
                    model = ols(f'Score ~ C(Time) * C(Q("{between}"))', data=long_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    # חישוב Eta Squared (גודל אפקט)
                    anova_table['Eta2'] = anova_table['sum_sq'] / (anova_table['sum_sq'].sum())
                    anova_table.columns = ['Sum of Squares', 'df', 'F', 'Sig.', 'Eta^2']
                    
                    # 3. תצוגת תוצאות ANOVA
                    st.write("### ANOVA Table (Within & Between Effects)")
                    st.table(anova_table.style.format("{:.3f}"))

                    # 4. גרף אינטראקציה (Interaction Plot)
                    st.write("### Descriptives Plot")
                    plot_data = long_df.groupby(['Time', between])['Score'].mean().reset_index()
                    fig = px.line(plot_data, x='Time', y='Score', color=between, markers=True, 
                                 line_shape='linear', template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    analysis_summary = f"RM ANOVA Results for {between}. Time p={anova_table.iloc[0,3]:.4f}, Interaction p={anova_table.iloc[2,3]:.4f}"
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- 2. מבחני T ומבחנים אחרים עם דו"ח מלא ---
    elif "t למדגמים בלתי תלויים" in test:
        with col_setup:
            dv = st.selectbox("משתנה תלוי:", numeric)
            iv = st.selectbox("משתנה בלתי תלוי (2 קבוצות):", all_cols)
            run = st.button("🚀 Run T-test")
        with col_results:
            if run:
                groups = [g[dv].dropna() for _, g in df.groupby(iv)]
                if len(groups) == 2:
                    # סטטיסטיקה תיאורית לקבוצות
                    st.write("### Group Statistics")
                    st.table(df.groupby(iv)[dv].agg(['count', 'mean', 'std']))
                    
                    # המבחן עצמו
                    t_stat, p = stats.ttest_ind(groups[0], groups[1])
                    st.write("### Independent Samples Test")
                    st.metric("Sig. (2-tailed)", f"{p:.4f}", delta="מובהק" if p < 0.05 else "לא מובהק")
                    st.write(f"t-value: {t_stat:.3f}")
                else: st.error("חייב 2 קבוצות בדיוק.")

    # (מקום למבחנים נוספים באותו פורמט עשיר)

    if analysis_summary:
        st.session_state['last_stat_result'] = analysis_summary
    if 'last_stat_result' in st.session_state:
        render_chat_interface(st.session_state['last_stat_result'])

def render_chat_interface(context):
    st.divider()
    st.subheader("🤖 צ'אט פירוש הממצאים (SPSS Mentor)")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("שאל אותי על ה-Eta Squared או על האינטראקציה..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                resp = model.generate_content(f"Analyze Context: {context}. User: {p}. Answer in Hebrew as a statistics professor.")
                st.markdown(resp.text)
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
            except Exception as e: st.error(f"AI Error: {e}")
