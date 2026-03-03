try:
        if uploaded.name.endswith('xlsx'):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
            
        # תיקון השגיאה: הפיכת כל כותרת לטקסט ואז ניקוי רווחים
        df.columns = [str(col).strip() for col in df.columns]
        
        # המרה למספרים וזיהוי עמודות רלוונטיות
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        cols = df_numeric.select_dtypes(include=['number']).columns.tolist()
        
        if not cols:
            st.error("לא נמצאו עמודות מספריות תקינות בקובץ.")
            st.stop()
            
    except Exception as e:
        st.error(f"שגיאה בטעינת הקובץ: {e}")
        st.stop()
