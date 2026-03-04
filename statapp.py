import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from modules.stats_lab import render_stats_lab
from ai_engine import render_ai_engine

st.set_page_config(page_title="Statistical Monster", layout="wide")

# פונקציה לטעינה חכמה - מזהה לבד איפה הכותרות
def load_data_smart(file):
    # קורא את 5 השורות הראשונות לבדיקה
    test_df = pd.read_excel(file, nrows=5, header=None)
    header_idx = 0
    for i, row in test_df.iterrows():
        # אם יש בשורה יותר טקסט ממספרים, זו כנראה הכותרת
        if row.apply(lambda x: isinstance(x, str)).sum() > len(test_df.columns) / 2:
            header_idx = i
            break
    file.seek(0)
    return pd.read_excel(file, header=header_idx)

# מנגנון איפוס בעת החלפת קובץ
uploaded_file = st.sidebar.file_uploader("העלה אקסל", type=["xlsx"])
if uploaded_file:
    if st.session_state.get('last_file') != uploaded_file.name:
        st.session_state.clear()
        st.session_state['last_file'] = uploaded_file.name
        st.rerun()

    df = load_data_smart(uploaded_file)
    
    # ניקוי שמות עמודות - קריטי למנועים סטטיסטיים
    df.columns = [str(c).strip().replace('.', '_').replace(' ', '_') for c in df.columns]
    
    # המרה אוטומטית: מה שמספרי הופך ל-float, מה שלא הופך ל-string
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > len(df) * 0.5: # אם רוב העמודה מספרים
            df[col] = converted
        else:
            df[col] = df[col].astype(str)

    # הצגת טאבים
    tab_stats, tab_ai = st.tabs(["🔬 מעבדה סטטיסטית", "🤖 Gemini AI"])
    with tab_stats:
        render_stats_lab(df)

