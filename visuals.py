import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import norm

def plot_normality_curve(data, col_name):
    # יצירת היסטוגרמה עם עקומת גאוס
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    p = norm.pdf(x, mu, std)

    fig = go.Figure()
    # היסטוגרמה
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Data Distribution', marker_color='#ABDDA4'))
    # עקומת גאוס
    fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Distribution', line=dict(color='red', width=3)))
    
    fig.update_layout(title=f"בדיקת נורמליות: {col_name} (Bell Curve)", template="plotly_white")
    return fig



def plot_ttest_results(res, col1_name, col2_name):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[col1_name, col2_name],
        y=[res['mean1'], res['mean2']],
        error_y=dict(type='data', array=[res['sd1']/(res['n']**0.5), res['sd2']/(res['n']**0.5)]),
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    fig.update_layout(title="השוואת ממוצעים עם Error Bars (SE)", template="plotly_white")
    return fig

def plot_heatmap(corr_df):
    return px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', title="מטריצת מתאמים")

def plot_regression(res, x_name, y_name):
    """גרף פיזור עם קו רגרסיה (Trendline)"""
    fig = px.scatter(x=res['x'], y=res['y'], labels={'x': x_name, 'y': y_name},
                     title=f"רגרסיה ליניארית: {y_name} מנובא ע'י {x_name}",
                     opacity=0.6)
    
    # הוספת קו המגמה
    x_range = np.linspace(res['x'].min(), res['x'].max(), 100)
    y_range = res['slope'] * x_range + res['intercept']
    
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', 
                             name=f"y = {res['slope']}x + {res['intercept']}",
                             line=dict(color='red', width=2)))
    
    fig.update_layout(template="plotly_white")
    return fig