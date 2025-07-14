import streamlit as st
import joblib
import pandas as pd

# Laad model (als je een model hebt)
clf = joblib.load("model_classification.pkl")  # Zorg ervoor dat het model bestand correct is

# Simuleer data in dit geval (geef echte data als je dat hebt)
data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6]
})

# Voorspelling doen met het model
prediction = clf.predict(data)

# Streamlit interface
st.title('Verzuim Dashboard')
st.write('Voorspelling:', prediction)
