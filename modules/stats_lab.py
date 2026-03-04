import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.markdown("""<style> .main { background-color: #f5f7f9; } .stTable { background-color: white; border-radius: 5px; } </style>""", unsafe_allow_html=True)
    
    # הכנת נתונים
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.title("🔬 Gemini Academic Stat-Suite")
    
    # סרגל כלים עליון סטייל SPSS
    menu = st.tabs(["📉 רגרסיות ומתאמים", "📈 ניתוח שונות (ANOVA)", "🛡️ אמינות ומדידה", "📊 תיאוריים"])

    # --- טור AI צדדי קבוע ---
    col_main, col_ai = st.columns([0.7, 0.3])

    with col_main:
        # --- TAB 1: REGRESSIONS & CORRELATIONS ---
        with menu[0]:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("הגדרות מבחן")
                m_type = st.radio("סוג מתאם/רגרסיה:", ["Pearson Correlation", "Spearman Rank", "Linear Regression"])
                vars_sel = st.multiselect("בחר משתנים:", numeric_cols)
                
                st.write("---")
                st.caption("אפשרויות נוספות:")
                show_scatter = st.checkbox("הצג דיאגרמת פיזור", value=True)
                show_coeffs = st.checkbox("הצג מקדמי רגרסיה (B, Beta)", value=True)

            with c2:
                if vars_sel:
                    if "Correlation" in m_type:
                        meth = "pearson" if "Pearson" in m_type else "spearman"
                        res = df[vars_sel].corr(method=meth)
                        st.write(f"### Matrix: {m_type}")
                        st.table(res.style.background_gradient(cmap='RdBu_r', axis=None).format("{:.3f}"))
                        if show_scatter and len(vars_sel) >= 2:
                            fig = px.scatter_matrix(df[vars_sel], height=400, template="plotly_white")
                            st.plotly_chart(fig)
                    
                    elif m_type == "Linear Regression" and len(vars_sel) >= 2:
                        y = df[vars_sel[0]]
                        X = sm.add_constant(df[vars_sel[1:]])
                        model = sm.OLS(y, X, missing='drop').fit()
                        st.write("### Model Summary (SPSS Style)")
                        st.text(model.summary2().as_text()) # פורמט טקסטואלי מקצועי

        # --- TAB 2: ANOVA & SIMPLE EFFECTS ---
        with menu[1]:
            st.subheader("Repeated Measures & Interaction")
            a1, a2 = st.columns([1, 2])
            with a1:
                dv_levels = st.multiselect("רמות זמן (Dependent):", numeric_cols, key="anova_dv")
                bs_factor = st.selectbox("גורם בין-נבדקי (Major):", all_cols, key="anova_bs")
                st.write("---")
                show_desc = st.checkbox("Descriptive Statistics", value=True)
                show_interaction = st.checkbox("Interaction Plot", value=True)
            
            with a2:
                if dv_levels and bs_factor and st.button("Execute ANOVA"):
                    # לוגיקת ANOVA מורחבת
                    tdf = df[dv_levels + [bs_factor]].dropna()
                    tdf['ID'] = range(len(tdf))
                    long = pd.melt(tdf, id_vars=['ID', bs_factor], value_vars=dv_levels, var_name='Time', value_name='Score')
                    long['Time'] = pd.Categorical(long['Time'], categories=dv_levels, ordered=True)
                    model = ols(f'Score ~ C(Time) * C(Q("{bs_factor}"))', data=long).fit()
                    table = sm.stats.anova_lm(model, typ=3)
                    st.table(table.style.format("{:.3f}"))
                    
                    if show_interaction:
                        
                        fig = px.line(long.groupby(['Time', bs_factor], observed=True)['Score'].mean().reset_index(), 
                                     x='Time', y='Score', color=bs_factor, markers=True)
                        st.plotly_chart(fig)

        # --- TAB 3: RELIABILITY ---
        with menu[2]:
            st.subheader("Scale Reliability Analysis")
            r1, r2 = st.columns([1, 2])
            with r1:
                items = st.multiselect("Items for Alpha:", numeric_cols, key="rel_items")
                st.checkbox("Item-Total Statistics")
                st.checkbox("Alpha if Item Deleted")
            with r2:
                if len(items) > 1:
                    idat = df[items].dropna()
                    k = len(items)
                    alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
                    st.metric("Cronbach's Alpha (α)", f"{alpha:.3f}")
                    

    # --- עמודת צ'אט (צד שמאל) ---
    with col_ai:
        st.subheader("🤖 AI Statistical Consultant")
        chat_container = st.container(height=600)
        with chat_container:
            if "messages" not in st.session_state: st.session_state.messages = []
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])
        
        if prompt := st.chat_input("שאל על התוצאות..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                st.chat_message("user").markdown(prompt)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt + " (Answer in Hebrew)")
                st.chat_message("assistant").markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
