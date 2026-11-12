"""4G Network Congestion Dashboard (simple version).

Three tabs:
1. Explore Data      - look at the dataset
2. Important Features - which inputs matter most to the model
3. Try It Yourself    - the trained model predicts the congestion type
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train_upd.csv"
MODEL_PATH = ROOT / "outputs" / "model.joblib"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
TARGET = "Congestion_Type"

st.set_page_config(page_title="4G Network Congestion Dashboard", layout="wide")
st.title("4G Network Congestion Analysis")
st.caption("Look at the data, see what matters, and try the trained model.")

if not TRAIN_PATH.exists():
    st.error("Training data not found. Keep data/train_upd.csv inside the project folder.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource(show_spinner="Preparing the model (about 30 seconds, first time only)...")
def load_model():
    """Load the saved model. If it is missing or was saved by a different
    scikit-learn version, train a fresh one here so the dashboard still works."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from train_model import make_pipeline, IDENTIFIER, USELESS_COLUMNS
        data = load_data()
        X = data.drop(columns=[TARGET, IDENTIFIER] + USELESS_COLUMNS, errors="ignore")
        y = data[TARGET].astype(str)
        model = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        return make_pipeline(model, X).fit(X, y)


df = load_data()
tab1, tab2, tab3 = st.tabs(["📊 Explore Data", "⭐ Important Features", "🤖 Try It Yourself"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (observations)", f"{len(df):,}")
    c2.metric("Input columns", str(df.shape[1] - 1))
    c3.metric("Congestion types", str(df[TARGET].nunique()))

    st.subheader("How many rows of each congestion type?")
    st.bar_chart(df[TARGET].value_counts())

    st.subheader("Rows by vendor (tower company)")
    st.bar_chart(df["ran_vendor"].value_counts())

    traffic_cols = [c for c in df.columns if c.endswith("_total_bytes")]
    st.subheader("Application traffic")
    selected = st.selectbox("Choose an application", traffic_cols)
    st.write(df[selected].describe().to_frame("Statistics"))

    st.subheader("Browse the records")
    pick = st.selectbox("Filter by congestion type", ["All"] + sorted(df[TARGET].unique()))
    view = df if pick == "All" else df[df[TARGET] == pick]
    st.dataframe(view.head(1000), width="stretch")
    st.caption("Showing at most 1,000 rows.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Which inputs matter most to the model?")
    st.write(
        "We shuffled each column one at a time and checked how much the model's accuracy "
        "dropped. A bigger drop means the model depends more on that column."
    )
    if IMPORTANCE_PATH.exists():
        imp = pd.read_csv(IMPORTANCE_PATH).head(15).set_index("feature")
        st.bar_chart(imp["importance"])
        st.caption("Top 15 columns. Many applications matter about equally, so no single "
                   "one decides the answer.")
    else:
        st.info("Run `python train_model.py` first to create this chart.")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Let the model predict the congestion type")
    model = load_model()
    columns = list(model.feature_names_in_)

    st.write("1. Pick a type of example.  2. Change the values if you like.  3. See the prediction.")
    left, right = st.columns([1, 1])
    with left:
        cls = st.selectbox("Start from an example of", sorted(df[TARGET].unique()))
        if "seed" not in st.session_state:
            st.session_state.seed = 0
        if st.button("Pick another example"):
            st.session_state.seed += 1
    seed = st.session_state.seed

    example = df[df[TARGET] == cls].sample(1, random_state=seed).iloc[0]
    row = example[columns].to_frame().T.infer_objects()

    top = []
    if IMPORTANCE_PATH.exists():
        top = [f for f in pd.read_csv(IMPORTANCE_PATH)["feature"] if f in columns][:5]

    with left:
        if top:
            st.write("**Change the 5 most important inputs:**")
        for feat in top:
            value = st.number_input(feat, min_value=0, value=int(example[feat]),
                                    key=f"{cls}_{seed}_{feat}")
            row[feat] = value

    prediction = model.predict(row)[0]
    probs = pd.Series(model.predict_proba(row)[0],
                      index=model.named_steps["model"].classes_, name="Probability")

    with right:
        st.metric("Model prediction", prediction)
        st.write("**How sure is the model?**")
        st.bar_chart(probs)
        if prediction == example[TARGET]:
            st.success(f"The real answer for this example is {example[TARGET]}.")
        else:
            st.warning(f"The real answer for this example was {example[TARGET]}.")
        st.caption("These examples come from the training data, so the model has seen "
                   "them before. Real accuracy on new data is about 79%.")