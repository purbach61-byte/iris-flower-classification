import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")
species = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements to predict the species.")

sl = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.8)
sw = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.0)
pl = st.number_input("Petal length (cm)", 1.0, 7.0, 4.3)
pw = st.number_input("Petal width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    pred = model.predict(np.array([[sl, sw, pl, pw]]))[0]
    st.success(f"Predicted species: **{species[pred]}**")
