import streamlit as st
import pandas as pd
import logic
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
st.write("העלה קובץ אקסל וקבל ניתוחים ותובנות AI")

uploaded_file = st.sidebar.file_uploader("העלה קובץ אקסל (XLSX)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("הקובץ נטען בהצלחה!")
    
    tab1, tab2, tab3 = st.tabs(["📋 נתונים גולמיים", "📈 ויזואליזציה", "🤖 ניתוח AI"])
    
    with tab1:
        st.subheader("מבט על הנתונים")
        st.write(df.head())
        st.subheader("סטטיסטיקה תיאורית")
        st.write(df.describe())
        
    with tab2:
        visuals.render_visuals(df)
        
    with tab3:
        ai_engine.render_ai_analysis(df)
else:
    st.info("ממתין להעלאת קובץ...")
