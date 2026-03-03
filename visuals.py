import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import norm

def plot_normality_curve(data, col_name):
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    p = norm.pdf(x, mu, std)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='התפלגות הנתונים', marker_color='#ABDDA4'))
    fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='עקומת נורמליות', line=dict(color='red', width=3)))
    fig.update_layout(title=f"בדיקת נורמליות: {col_name}", template="plotly_white")
    return fig

# פונקציית הקישור ש-statapp.py מחפש
def render_visuals(df):
    st.subheader("📊 ניתוח ויזואלי מתקדם")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.warning("לא נמצאו עמודות מספריות לניתוח.")
        return

    # בחירת סוג גרף
    viz_type = st.selectbox("בחר סוג ניתוח", ["התפלגות ונורמליות", "מטריצת מתאמים (Heatmap)", "פיזור ורגרסיה"])

    if viz_type == "התפלגות ונורמליות":
        col = st.selectbox("בחר עמודה לבדיקה", numeric_cols)
        fig = plot_normality_curve(df[col].dropna(), col)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "מטריצת מתאמים (Heatmap)":
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="מטריצת מתאמים בין המשתנים")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("צריך לפחות 2 עמודות מספריות למטריצת מתאמים.")

    elif viz_type == "פיזור ורגרסיה":
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("בחר משתנה מנבא (X)", numeric_cols)
            col_y = st.selectbox("בחר משתנה תלוי (Y)", numeric_cols)
            fig = px.scatter(df, x=col_x, y=col_y, trendline="ols", title=f"קשר בין {col_x} ל-{col_y}")
            st.plotly_chart(fig, use_container_width=True)
