import streamlit as st
import google.generativeai as genai

def render_ai_analysis(df):
    st.subheader("🤖 עוזר מחקר סטטיסטי אישי (Gemini 2.0)")

    # 1. אתחול הזיכרון (Chat History) אם הוא לא קיים עדיין
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. הכנת מידע על בסיס הנתונים (כדי שה-AI יכיר את השמות החדשים)
    column_names = ", ".join(df.columns.tolist())
    stats_preview = df.describe(include='all').to_string()

    # הגדרת הנחיות המערכת (System Instruction) - מי ה-AI ומה הוא יודע
    system_instruction = f"""
    אתה עוזר מחקר וסטטיסטיקאי מקצועי.
    שמות המשתנים בקובץ של המשתמש הם: {column_names}
    להלן סיכום סטטיסטי של הנתונים:
    {stats_preview}
    
    הנחיות חשובות:
    - ענה תמיד בעברית.
    - התייחס לשמות המשתנים בדיוק כפי שהם מופיעים (למשל 'Major', 'Pre Score').
    - שמור על הקשר של השיחה (זכור מה המשתמש אמר קודם).
    - הצע מבחנים סטטיסטיים מתאימים (t-test, ANOVA, רגרסיה) בהתאם לשאלות.
    """

    # 3. הצגת היסטוריית ההתכתבות על המסך
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. תיבת קלט לצ'אט (כמו ב-ChatGPT)
    if prompt := st.chat_input("שאל אותי על הנתונים שלך..."):
        # הצגת הודעת המשתמש
        st.chat_message("user").markdown(prompt)
        # שמירה להיסטוריה
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # קריאה למודל Gemini 2.0
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # יצירת הניתוח כולל ה-Context וההיסטוריה
            # אנחנו שולחים את ההנחיות יחד עם ההיסטוריה
            full_prompt = f"{system_instruction}\n\nהיסטוריית שיחה:\n"
            for m in st.session_state.messages:
                full_prompt += f"{m['role']}: {m['content']}\n"
            
            with st.spinner("חושב..."):
                response = model.generate_content(full_prompt)
                ai_response = response.text

            # הצגת תשובת ה-AI
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            
            # שמירת תשובת ה-AI להיסטוריה
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            st.error(f"אירעה שגיאה בתקשורת עם ה-AI: {e}")

    # כפתור לאיפוס השיחה (בצד)
    if st.button("נקה היסטוריית שיחה"):
        st.session_state.messages = []
        st.rerun()

