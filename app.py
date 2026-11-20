import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")

model = joblib.load("iris_model.pkl")
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
species = ["Setosa", "Versicolor", "Virginica"]
images = ["Iris_setosa.jpg", "Iris_versicolor.jpg", "Iris_virginica.jpg"]

st.title("🌸 Iris Flower Classification")
st.caption("Decision Tree Classifier | Iris dataset")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Predict", "Dataset Overview", "EDA", "Model Evaluation"])

# ---------- Tab 1: Prediction ----------
with tab1:
    st.write("Enter the flower measurements to predict the species.")
    sl = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.8)
    sw = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.0)
    pl = st.number_input("Petal length (cm)", 1.0, 7.0, 4.3)
    pw = st.number_input("Petal width (cm)", 0.1, 2.5, 1.3)

    if st.button("Predict"):
        pred = model.predict(np.array([[sl, sw, pl, pw]]))[0]
        st.success(f"Predicted species: **{species[pred]}**")
        try:
            st.image(images[pred], caption=f"Iris {species[pred]}", width=300)
        except Exception:
            st.warning("Flower image not found.")

    st.divider()
    st.subheader("Model Information")
    st.write("**Algorithm used:**", type(model).__name__)
    st.write("**Tree depth:**", model.get_depth())
    st.write("**Number of leaves:**", model.get_n_leaves())
    st.write("**Number of features:**", model.n_features_in_)

# ---------- Tab 2: Dataset Overview ----------
with tab2:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", df.shape[0])
    c2.metric("Features", 4)
    c3.metric("Classes", 3)
    st.write("**First five records**")
    st.dataframe(df.head())
    st.write("**Statistical summary**")
    st.dataframe(df.drop(columns="target").describe())
    st.write("**Missing values (preprocessing check)**")
    st.dataframe(df.isnull().sum().rename("missing"))
    st.write("**Records per species**")
    st.dataframe(df["species"].value_counts())

# ---------- Tab 3: EDA ----------
with tab3:
    st.subheader("Exploratory Data Analysis")
    st.write("**Pairplot of features by species**")
    grid = sns.pairplot(df.drop(columns="target"), hue="species")
    st.pyplot(grid.figure)

    st.write("**Correlation heatmap**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.drop(columns=["target", "species"]).corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------- Tab 4: Model Evaluation ----------
with tab4:
    st.subheader("Model Evaluation")
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    st.write(f"Training samples: **{len(X_train)}** | "
             f"Testing samples: **{len(X_test)}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.3f}")
    m4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

    st.write("**Classification report**")
    report = classification_report(
        y_test, y_pred, target_names=iris.target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.write("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("**Decision tree structure**")
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(model, feature_names=iris.feature_names,
              class_names=list(iris.target_names), filled=True, ax=ax)
    st.pyplot(fig)
