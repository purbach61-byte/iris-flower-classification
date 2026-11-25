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

st.set_page_config(page_title="Floria | Iris Classification",
                   page_icon="🌸", layout="wide")

# ---------------------------------------------------------------- styling
BG, CARD, LINE = "#050308", "#100b17", "#2a2236"
TEXT, MUTED, ACCENT, ROSE = "#f3edf7", "#9a90a8", "#a78bfa", "#e58aa8"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background: radial-gradient(1200px 600px at 80% -10%, #1d1030 0%, #050308 60%); color: #f3edf7; font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: #f3edf7 !important; font-weight: 700 !important; }
p, label, span, div { font-family: 'Inter', sans-serif; }
p, li, label { color: #cfc6dc; }

/* brand + tabs as navigation */
.brand { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: #f3edf7; margin-bottom: .2rem; }
.brand span { color: #a78bfa; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #2a2236; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #9a90a8; font-size: .95rem; letter-spacing: .06em; padding: .6rem 0; }
.stTabs [aria-selected="true"] { color: #f3edf7 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #f3edf7; }

/* hero */
.hero-title { font-family: 'Playfair Display', serif; font-size: 4.6rem; line-height: 1.02; color: #f3edf7; margin: 1.5rem 0 1rem 0; }
.hero-title em { font-weight: 500; color: #e9dff5; }
.hero-text { font-size: 1.08rem; line-height: 1.7; color: #b3a9c2; max-width: 34rem; }
.pills { display: flex; flex-wrap: wrap; gap: .7rem; margin-top: 1.6rem; }
.pill { border: 1px solid #2a2236; background: #0b0710; border-radius: 999px; padding: .5rem 1rem; font-size: .8rem; color: #9a90a8; }
.pill b { color: #f3edf7; font-weight: 600; margin-left: .35rem; }

/* feature cards */
.fcard { border: 1px solid #2a2236; background: linear-gradient(180deg, #14101c 0%, #0b0710 100%); border-radius: 18px; padding: 1.4rem; height: 100%; }
.fcard.root { border-color: #6b3b57; background: linear-gradient(180deg, #1d1020 0%, #0b0710 100%); }
.badge { display: inline-block; border: 1px solid #2a2236; border-radius: 999px; padding: .25rem .7rem; font-size: .68rem; color: #9a90a8; letter-spacing: .08em; text-transform: uppercase; }
.fcard.root .badge { border-color: #8a4a6b; color: #e58aa8; }
.fcard h3 { font-size: 1.35rem; margin: 1rem 0 .5rem 0; }
.fcard p { font-size: .88rem; line-height: 1.6; color: #a99fb8; min-height: 5.6rem; }
.frow { display: flex; justify-content: space-between; font-size: .82rem; padding: .35rem 0; border-top: 1px solid #1d1727; }
.frow span:first-child { color: #7d7390; }
.frow span:last-child { color: #f3edf7; font-weight: 500; }

/* result card */
.result { border: 1px solid #6b3b57; background: linear-gradient(180deg, #1d1020 0%, #0b0710 100%); border-radius: 18px; padding: 1.4rem 1.6rem; margin: 1rem 0; }
.result small { color: #9a90a8; letter-spacing: .08em; text-transform: uppercase; font-size: .7rem; }
.result .name { font-family: 'Playfair Display', serif; font-size: 2.4rem; color: #f3edf7; }

/* inputs, buttons, metrics */
.stNumberInput input { background: #0b0710 !important; color: #f3edf7 !important; border-radius: 10px; }
.stNumberInput > div > div { border-radius: 10px; border-color: #2a2236 !important; }
.stButton button { background: #a78bfa; color: #07050b; border: none; border-radius: 999px; padding: .65rem 2rem; font-weight: 600; }
.stButton button:hover { background: #c4b5fd; color: #07050b; }
.stButton button:focus, .stButton button:active { background: #a78bfa; color: #07050b; }
div[data-testid="stMetric"] { border: 1px solid #2a2236; background: #0b0710; border-radius: 14px; padding: 1rem 1.2rem; }
div[data-testid="stMetricLabel"] p { color: #9a90a8; }
div[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; color: #f3edf7; }
div[data-testid="stImage"] img { border-radius: 16px; }
hr { border-color: #2a2236; }
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { font-family: 'Playfair Display', serif !important; }
</style>
""", unsafe_allow_html=True)

# charts in the same dark palette
plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": LINE, "grid.color": LINE,
})

# ------------------------------------------------------------------- data
SPECIES = ["Setosa", "Versicolor", "Virginica"]
IMAGES = ["Iris_setosa.jpg", "Iris_versicolor.jpg", "Iris_virginica.jpg"]


@st.cache_resource
def load_model():
    return joblib.load("iris_model.pkl")


@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return iris, df


model = load_model()
iris, df = load_data()
feature_names = list(iris.feature_names)

# Same split as train_model.py, so the numbers match the training run
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------------------------------ layout
st.markdown('<div class="brand">Floria<span>.</span></div>', unsafe_allow_html=True)
tab_overview, tab_features, tab_classify, tab_insights = st.tabs(
    ["Overview", "Features", "Classify", "Insights"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""
        <div class="hero-title">Botanic<br><em>Architecture.</em></div>
        <div class="hero-text">A machine learning system that classifies Iris flowers
        into three species, Setosa, Versicolor and Virginica, from their sepal and
        petal measurements using a Decision Tree.</div>
        <div class="pills">
          <div class="pill">Algorithm<b>{type(model).__name__}</b></div>
          <div class="pill">Dataset<b>{len(df)} observations</b></div>
          <div class="pill">Test accuracy<b>{accuracy*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        cols = st.columns(3)
        for col, name, img in zip(cols, SPECIES, IMAGES):
            with col:
                try:
                    st.image(img, caption=f"Iris {name}", use_container_width=True)
                except Exception:
                    st.caption(f"Iris {name} (image missing)")

# ---------------------------------------------------------------- Features
with tab_features:
    st.markdown("<h2 style='text-align:center;margin-top:1.5rem'>Four botanical features</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;max-width:42rem;margin:0 auto 2rem auto'>"
                "The Decision Tree weighs four measurements in centimetres. "
                "Its root question and the importance of each feature come "
                "directly from the trained model.</p>", unsafe_allow_html=True)

    tree = model.tree_
    root_feature = int(tree.feature[0])
    root_threshold = float(tree.threshold[0])
    importances = model.feature_importances_
    setosa_sw = df.loc[df["species"] == "setosa", "sepal width (cm)"].mean()

    blurbs = [
        "The length of the outer protective leaf that encloses the flower bud.",
        f"The width of the sepal. Setosa has wider sepals (mean {setosa_sw:.2f} cm) than Versicolor.",
        f"The tree's first question: petal length at most {root_threshold:.2f} cm separates Setosa perfectly.",
        "Used in the branch splits that separate Versicolor from Virginica.",
    ]
    titles = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

    cols = st.columns(4, gap="medium")
    for i, col in enumerate(cols):
        fname = feature_names[i]
        is_root = i == root_feature
        if is_root:
            badge = "Root decision split"
        elif importances[i] > 0.01:
            badge = "Branch decision split"
        else:
            badge = "Calyx dimension"
        with col:
            st.markdown(f"""
            <div class="fcard {'root' if is_root else ''}">
              <span class="badge">{badge}</span>
              <h3>{titles[i]}</h3>
              <p>{blurbs[i]}</p>
              <div class="frow"><span>Range</span><span>{df[fname].min():.1f} to {df[fname].max():.1f} cm</span></div>
              <div class="frow"><span>Mean</span><span>{df[fname].mean():.2f} cm</span></div>
              <div class="frow"><span>Importance</span><span>{importances[i]*100:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------- Classify
with tab_classify:
    st.markdown("<h2 style='margin-top:1.5rem'>Classify a flower</h2>", unsafe_allow_html=True)
    st.write("Enter the four measurements and the Decision Tree predicts the species.")

    c1, c2 = st.columns(2)
    sl = c1.number_input("Sepal length (cm)", 4.0, 8.0, 5.8)
    sw = c2.number_input("Sepal width (cm)", 2.0, 4.5, 3.0)
    pl = c1.number_input("Petal length (cm)", 1.0, 7.0, 4.3)
    pw = c2.number_input("Petal width (cm)", 0.1, 2.5, 1.3)

    if st.button("Classify flower"):
        sample = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)
        pred = int(model.predict(sample)[0])
        conf = float(model.predict_proba(sample)[0][pred])
        r1, r2 = st.columns([1.4, 1], gap="large")
        with r1:
            st.markdown(f"""
            <div class="result">
              <small>Predicted species</small>
              <div class="name">Iris {SPECIES[pred]}</div>
              <p style="margin:.4rem 0 0 0">Leaf confidence: {conf*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            try:
                st.image(IMAGES[pred], caption=f"Iris {SPECIES[pred]}", width=300)
            except Exception:
                st.warning("Flower image not found.")

    st.divider()
    st.markdown("<h3>Model information</h3>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", "Decision Tree")
    m2.metric("Tree depth", model.get_depth())
    m3.metric("Leaves", model.get_n_leaves())
    m4.metric("Features", model.n_features_in_)
    st.caption(f"Model class: {type(model).__name__}")

# ---------------------------------------------------------------- Insights
with tab_insights:
    st.markdown("<h2 style='margin-top:1.5rem'>Insights</h2>", unsafe_allow_html=True)

    st.markdown("<h3>Dataset overview</h3>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Records", df.shape[0])
    d2.metric("Features", 4)
    d3.metric("Classes", 3)
    st.write("**First five records**")
    st.dataframe(df.head(), use_container_width=True)
    st.write("**Statistical summary**")
    st.dataframe(df.drop(columns="target").describe(), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.write("**Missing values (preprocessing check)**")
        st.dataframe(df.isnull().sum().rename("missing"), use_container_width=True)
    with b:
        st.write("**Records per species**")
        st.dataframe(df["species"].value_counts(), use_container_width=True)

    st.divider()
    st.markdown("<h3>Exploratory data analysis</h3>", unsafe_allow_html=True)
    st.write("**Pairplot of features by species**")
    grid = sns.pairplot(df.drop(columns="target"), hue="species",
                        palette=["#a78bfa", "#e58aa8", "#f2c879"])
    st.pyplot(grid.figure)

    st.write("**Correlation heatmap**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.drop(columns=["target", "species"]).corr(),
                annot=True, cmap="magma", ax=ax)
    st.pyplot(fig)

    st.divider()
    st.markdown("<h3>Model evaluation</h3>", unsafe_allow_html=True)
    st.write(f"Training samples: **{len(X_train)}** | Testing samples: **{len(X_test)}**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", f"{accuracy:.3f}")
    e2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.3f}")
    e3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.3f}")
    e4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

    st.write("**Classification report**")
    report = classification_report(y_test, y_pred,
                                   target_names=iris.target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.write("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="magma",
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("**Decision tree structure**")
    fig, ax = plt.subplots(figsize=(12, 7))
    artists = plot_tree(model, feature_names=feature_names,
                        class_names=list(iris.target_names), filled=True, ax=ax)
    for a in artists:
        if a.get_bbox_patch() is not None:
            a.set_color("#111111")
    st.pyplot(fig)
