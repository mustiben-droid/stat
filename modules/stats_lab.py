import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols

def render_stats_lab(df: pd.DataFrame):
    # שינוי השם לבקשתך
    st.header("🔬 Statistic Analyzer")
    
    # ניקוי בסיסי של שמות העמודות
    df.columns = df.columns.str.strip()
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # תפריט ניתוחים ראשי (Sidebar)
    analysis_type = st.sidebar.radio("Analysis Menu", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🧪 T-Tests",
        "🔲 Frequencies (Chi-Square)",
        "🔗 Regression & Correlation"
    ])

    st.divider()

    # --- 1. DESCRIPTIVES (כולל SE ו-CV) ---
    if analysis_type == "📊 Descriptives":
        col1, col2 = st.columns([1, 2])
        with col1:
            vars_to_analyze = st.multiselect("Variables", numeric)
            split_by = st.selectbox("Split by (Grouping Variable)", ["None"] + all_cols)
            st.subheader("Options")
            show_se = st.checkbox("Std. error mean", value=True)
            show_cv = st.checkbox("Coefficient of variation", value=True)
        
        with col2:
            if vars_to_analyze:
                st.write("### Descriptive Statistics")
                if split_by == "None":
                    res = df[vars_to_analyze].agg(['count', 'mean', 'std', 'min', 'max']).T
                else:
                    res = df.groupby(split_by)[vars_to_analyze].agg(['count', 'mean', 'std']).stack(level=0)
                
                if show_se:
                    res['SE'] = res['std'] / np.sqrt(res['count'])
                if show_cv:
                    res['CV'] = res['std'] / res['mean']
                
                st.table(res.style.format("{:.3f}"))

    # --- 2. REPEATED MEASURES ANOVA (הפלט המלא שביקשת) ---
    elif analysis_type == "📈 ANOVA (Repeated Measures)":
        col_setup, col_out = st.columns([1, 2])
        with col_setup:
            levels = st.multiselect("RM Factors (Levels)", numeric, help="Select levels in order (e.g., Level 1, 2, 3)")
            between_subject = st.selectbox("Between Subjects Factor (Major)", all_cols)
            
            st.subheader("Options")
            display_plots = st.checkbox("Descriptive plots", value=True)
            est_effect_size = st.checkbox("Estimates of effect size (η²p)", value=True)
            simple_main = st.checkbox("Simple main effects", value=True)

        with col_out:
            if len(levels) > 1:
                try:
                    # הכנת נתונים לפורמט ארוך
                    temp_df = df[levels + [between_subject]].dropna().copy()
                    temp_df['ID'] = range(len(temp_df))
                    long_df = pd.melt(temp_df, id_vars=['ID', between_subject], value_vars=levels, 
                                     var_name='Time', value_name='Score')
                    long_df['Time'] = pd.Categorical(long_df['Time'], categories=levels, ordered=True)

                    # א. טבלת ANOVA (Within-Subjects)
                    st.write("### Tests of Within-Subjects Effects")
                    model = ols(f'Score ~ C(Time) * C(Q("{between_subject}"))', data=long_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=3) # Type III SS כמקובל ב-JASP
                    
                    # הוספת Mean Square ו-Eta Squared
                    anova_table['Mean Square'] = anova_table['sum_sq'] / anova_table['df']
                    if est_effect_size:
                        ss_resid = anova_table.loc['Residual', 'sum_sq']
                        anova_table['η²p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_resid)
                        anova_table.at['Residual', 'η²p'] = np.nan

                    # עיצוב הטבלה (צביעת ערכי p מובהקים)
                    st.table(anova_table.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))

                    # ב. Simple Main Effects (לפי כל רמה בנפרד)
                    if simple_main:
                        st.write("### Simple Main Effects")
                        sme_res = []
                        for lvl in levels:
                            m = ols(f'Q("{lvl}") ~ C(Q("{between_subject}"))', data=temp_df).fit()
                            a = sm.stats.anova_lm(m, typ=3)
                            sme_res.append({
                                "Level of Factor": lvl, 
                                "Sum of Squares": a.iloc[1,0],
                                "df": a.iloc[1,1],
                                "F": a.iloc[1,2], 
                                "p": a.iloc[1,3]
                            })
                        st.table(pd.DataFrame(sme_res).style.format({"Sum of Squares": "{:.3f}", "F": "{:.3f}", "p": "{:.4f}"}))

                    # ג. גרפים (Plots)
                    if display_plots:
                        st.write("### Descriptives Plots")
                        plot_data = long_df.groupby(['Time', between_subject], observed=True)['Score'].mean().reset_index()
                        fig = px.line(plot_data, x='Time', y='Score', color=between_subject, markers=True, template="plotly_white")
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error during ANOVA calculation: {e}")

    # --- 3. CHI-SQUARE ---
    elif analysis_type == "🔲 Frequencies (Chi-Square)":
        v1 = st.selectbox("Rows (Variable 1)", all_cols)
        v2 = st.selectbox("Columns (Variable 2)", all_cols)
        if st.button("Run Contingency Table"):
            ct = pd.crosstab(df[v1], df[v2])
            st.write("### Contingency Table")
            st.table(ct)
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.metric("Pearson Chi-Square Sig.", f"{p:.4f}")

# בתוך פונקציית render_chat_interface או בסוף render_stats_lab:

if prompt := st.chat_input("שאל אותי על תוצאות הניתוח..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # עדכון המודל לגרסה החדשה ביותר
            model = genai.GenerativeModel('gemini-2-flash') 
            
            # שליפת ההקשר הסטטיסטי האחרון מה-ANOVA או מה-Descriptives
            context = st.session_state.get('last_context', "No analysis run yet.")
            
            full_prompt = f"""
            Context: {context}
            User Question: {prompt}
            Instructions: You are a senior statistical consultant. 
            Analyze the data provided in the context. 
            Answer in Hebrew. 
            If there is a decline (נסיגה), point it out specifically.
            """
            
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            # אם ה-API עדיין לא מעודכן ל-3 בסביבה שלך, הוא יחזור ל-1.5 אוטומטית כדי לא לקרוס
            st.error(f"Error: {e}")
