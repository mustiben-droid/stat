import streamlit as st
import google.generativeai as genai
import plotly.express as px

def render_ai_analysis(df):
    st.subheader("🤖 עוזר מחקר וניתוח ויזואלי")

    # --- חלק 1: בונה גרפים מהיר (SPSS Style) ---
    with st.expander("📊 בונה גרפים מהיר וניתוח אוטומטי", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("סוג גרף", ["גרף פיזור", "גרף עמודות (ממוצעים)", "תיבת נתונים (Boxplot)"])
        with col2:
            x_var = st.selectbox("ציר X (משתנה בלתי תלוי)", df.columns)
        with col3:
            y_var = st.selectbox("ציר Y (משתנה תלוי)", df.select_dtypes(include=['number']).columns)

        if st.button("צור גרף ונתח"):
            if chart_type == "גרף פיזור":
                fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", template="plotly_white")
            elif chart_type == "גרף עמודות (ממוצעים)":
                fig = px.bar(df, x=x_var, y=y_var, barmode="group", template="plotly_white")
            else:
                fig = px.box(df, x=x_var, y=y_var, template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # שליחה אוטומטית ל-AI לניתוח הגרף שנוצר
            st.info("💡 ניתוח ה-AI לגרף שנוצר:")
            with st.spinner("מנתח את המגמות..."):
                model = genai.GenerativeModel('gemini-2.0-flash')
                analysis_prompt = f"נתח בקצרה ובשורה תחתונה את הקשר בין {x_var} ל-{y_var} בגרף {chart_type}. הנה הסטטיסטיקה: {df[[x_var, y_var]].describe().to_string()}"
                response = model.generate_content(analysis_prompt)
                st.write(response.text)

    st.divider()

    # --- חלק 2: הצ'אט המוכר (עם היסטוריה) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("שאל אותי על הנתונים או בקש המלצה לגרף..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            column_names = ", ".join(df.columns.tolist())
            
            full_context = f"""
            אתה עוזר מחקר סטטיסטי. השתמש בשמות המשתנים: {column_names}.
            ענה בקצרה ובתכל'ס. אם המשתמש שואל על קשרים, הצע לו איזה גרף לבנות ב'בונה הגרפים' למעלה.
            היסטוריה: {st.session_state.messages[-3:]} 
            שאלה: {prompt}
            """
            
            response = model.generate_content(full_context)
            with st.chat_message("assistant"):
                st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"שגיאה: {e}")
