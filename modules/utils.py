import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# --- עזרי AI ---

def ai_show(prompt):
    """מציג תשובה מ-Gemini בתוך תיבה מעוצבת"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        st.info("🤖 **פרשנות AI:**")
        st.write(response.text)
    except Exception as e:
        st.error(f"שגיאת AI: {e}")

def store_result(res_str):
    """שומר את התוצאה האחרונה בזיכרון האפליקציה"""
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = []
    st.session_state['last_result'].append(res_str)

# --- עזרי עיצוב וטבלאות ---

def apa_table(rows):
    """מציג טבלה פשוטה במראה נקי של APA"""
    df_apa = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.table(df_apa)

def significance_badge(p):
    """מחזיר אינדיקציה ויזואלית למובהקות"""
    if p < 0.001: return "Significant (p < .001) ✅✅"
    if p < 0.01:  return "Significant (p < .01) ✅"
    if p < 0.05:  return "Significant (p < .05) ✅"
    return "Not Significant (p > .05) ❌"

# --- עזרי סטטיסטיקה ---

def label_eta(eta2):
    """פרשנות לגודל אפקט מסוג אטה בריבוע"""
    if eta2 < 0.01: return "Small"
    if eta2 < 0.06: return "Medium"
    return "Large"

def label_r(r):
    """פרשנות לעוצמת מתאם"""
    r_abs = abs(r)
    if r_abs < 0.3: return "Weak"
    if r_abs < 0.5: return "Moderate"
    return "Strong"

def bonferroni_posthoc(df, gv, dv):
    """חישוב פוסט-הוק בונפרוני בסיסי בין קבוצות"""
    from itertools import combinations
    import scipy.stats as stats
    
    groups = df[gv].unique()
    pairs = list(combinations(groups, 2))
    num_tests = len(pairs)
    
    results = []
    for g1, g2 in pairs:
        x = df[df[gv] == g1][dv].dropna()
        y = df[df[gv] == g2][dv].dropna()
        t_stat, p = stats.ttest_ind(x, y)
        adj_p = min(p * num_tests, 1.0)
        results.append({
            "Comparison": f"{g1} vs {g2}",
            "p-value": f"{p:.4f}",
            "Adjusted p": f"{adj_p:.4f}",
            "Sig": "✅" if adj_p < 0.05 else "❌"
        })
    return pd.DataFrame(results)
