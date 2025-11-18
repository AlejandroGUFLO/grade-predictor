
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Load data and train model
# ------------------------------
df = pd.read_csv("proyectom.csv")

df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)

X = df[[
    "Materias pasadas ",
    "Materias nuevas",
    "Horas estudio pasadas ",
    "Horas de estudio actuales ",
    "Calificaciones pasadas"
]]

Y = df["HighPerformance"]

model = LogisticRegression()
model.fit(X, Y)

# ------------------------------
# UI – Inputs
# ------------------------------
st.title("🎓 Academic Performance Predictor")

st.write("Enter your academic data to predict if your performance will be above 9.2")

gender = st.selectbox("Gender", ["Masculino", "Femenino"])

semester = st.selectbox("Current semester", list(range(1, 10)))

courses_past = st.number_input("Courses taken last semester", min_value=0.0)
hours_past = st.number_input("Study hours per week last semester", min_value=0.0)
grade_past = st.number_input("Final grade last semester", min_value=0.0, max_value=10.0)

courses_now = st.number_input("Courses taken now", min_value=0.0)
hours_now = st.number_input("Study hours per week now", min_value=0.0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    new_data = pd.DataFrame({
        "Materias pasadas ": [courses_past],
        "Materias nuevas": [courses_now],
        "Horas estudio pasadas ": [hours_past],
        "Horas de estudio actuales ": [hours_now],
        "Calificaciones pasadas": [grade_past]
    })

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    st.subheader("📌 Prediction Result")
    st.write(f"**High Performance:** {'Sí (≥9.2)' if prediction == 1 else 'No (<9.2)'}")
    st.write(f"**Probability:** {probability:.2f}")
