import streamlit as st
import joblib
import pandas as pd

# Laad model en controleer
try:
    clf = joblib.load("model_classification.pkl")
    st.write("Model geladen succesvol.")
except Exception as e:
    st.write(f"Fout bij het laden van het model: {e}")

# Simuleer data als fallback (voeg echte data toe indien beschikbaar)
data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6]
})

# Voorspelling doen met het model als het goed geladen is
if clf:
    try:
        prediction = clf.predict(data)
        st.write('Voorspelling:', prediction)
    except Exception as e:
        st.write(f"Fout bij het uitvoeren van de voorspelling: {e}")

# Interface
st.title('Verzuim Dashboard')
st.write("Hier kun je de voorspelling van het verzuim op basis van het model zien.")
