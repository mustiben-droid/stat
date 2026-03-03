import streamlit as st
import pandas as pd
from .utils import ai_show

def render_consultation(df: pd.DataFrame):
    st.header("💬 צ'אט ייעוץ סטטיסטי")
    st.write("יש לך שאלה על הנתונים? לא בטוח איך להמשיך? התייעץ עם ה-AI המומחה שלנו.")

    # הצגת מידע על הנתונים שה-AI "רואה"
    with st.expander("👁️ מה ה-AI יודע על הנתונים שלך?"):
        summary = {
            "columns": df.columns.tolist(),
            "sample_size": len(df),
            "numeric_vars": df.select_dtypes(include='number').columns.tolist(),
            "nominal_vars": df.select_dtypes(exclude='number').columns.tolist()
        }
        st.json(summary)

    # ממשק הצ'אט
    user_question = st.text_area("שאל שאלה (למשל: 'האם המשתנים שלי מתאימים לרגרסיה?' או 'מה המשמעות של p-value נמוך?')", 
                                placeholder="כתוב כאן את שאלתך...")

    if st.button("🚀 שלח שאלה"):
        if user_question:
            # בניית פרומפט חכם שכולל את ההקשר של הנתונים
            context_prompt = f"""
            אתה יועץ סטטיסטי בכיר המסייע לסטודנטים בתזה. 
            בסיס הנתונים הנוכחי כולל את המשתנים הבאים: {', '.join(df.columns.tolist())}.
            גודל המדגם הוא {len(df)} נבדקים.
            המשתנים המספריים הם: {', '.join(summary['numeric_vars'])}.
            
            השאלה של הסטודנט: {user_question}
            
            ענה בצורה מקצועית, מעודדת וברורה. אם השאלה נוגעת לניתוח ספציפי, הסבר את הלוגיקה הסטטיסטית מאחוריו.
            """
            ai_show(context_prompt)
        else:
            st.warning("אנא כתוב שאלה לפני השליחה.")

    st.divider()
    st.info("💡 טיפ: ה-AI מכיר את שמות המשתנים שלך, אז אתה יכול לשאול שאלות ספציפיות כמו 'איזה קשר כדאי לבדוק בין משתנה X למשתנה Y?'")
