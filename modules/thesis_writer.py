import streamlit as st
from .utils import ai_show

def render_thesis_writer():
    st.header("✍️ עוזר כתיבת התזה (APA Writer)")
    st.write("כאן תוכל להפוך את הממצאים הסטטיסטיים שלך לפסקת ממצאים בפורמט APA.")

    # בדיקה אם יש תוצאות שמורות בזיכרון
    results = st.session_state.get('last_result', [])

    if not results:
        st.info("עדיין לא הרצת ניתוחים במעבדה. הרץ ניתוח (כמו ANOVA או רגרסיה) כדי שנוכל לעזור לך לכתוב אותו.")
    else:
        st.subheader("📋 ממצאים שנאספו מהמעבדה:")
        for i, res in enumerate(results):
            st.code(f"ניתוח {i+1}: {res}")

        st.divider()
        
        st.subheader("🤖 יצירת פרק ממצאים")
        research_goal = st.text_area("מה הייתה השערת המחקר שלך?", 
                                   placeholder="למשל: נבדק הקשר בין שעות שינה לציונים במבחן...")
        
        if st.button("🪄 נסח לי פסקת ממצאים"):
            all_res_str = " | ".join(results)
            prompt = f"""
            אתה מומחה לכתיבה אקדמית בפורמט APA. 
            מטרת המחקר: {research_goal}
            התוצאות הסטטיסטיות: {all_res_str}
            
            משימה:
            כתוב פסקת ממצאים אקדמית בעברית המדווחת על הממצאים הללו.
            דגשים:
            1. השאר מונחים סטטיסטיים (p, F, t, R2, SD, M) באנגלית.
            2. השתמש בשפה מדעית ומדויקת.
            3. ציין האם ההשערה אוששה או נדחתה לפי התוצאות.
            """
            ai_show(prompt)

    # כפתור ניקוי
    if st.button("🗑️ נקה היסטוריית תוצאות"):
        st.session_state['last_result'] = []
        st.rerun()
