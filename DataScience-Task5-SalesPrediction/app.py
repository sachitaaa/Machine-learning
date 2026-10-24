
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sales_prediction_pipeline.pkl"
DATA_PATH = BASE_DIR / "Advertising (1).csv"

st.title("📊 Advertising Sales Prediction")
st.write(
    "Explore how TV, radio, and newspaper advertising budgets "
    "relate to sales, and predict sales using a trained Random Forest model."
)

# -----------------------------
# Load model and dataset
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.stop()

model = load_model()
df = load_data()

features = ["TV", "radio", "newspaper"]
X = df[features]
y = df["sales"]

# Evaluate on a held-out test set.
# This assumes the same 80/20 split and random_state used in your notebook.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# -----------------------------
# Performance metrics
# -----------------------------
st.subheader("📈 Model Performance on Test Data")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.3f}", help="Lower is better.")
c2.metric("RMSE", f"{rmse:.3f}", help="Lower is better; penalizes larger errors.")
c3.metric("R² Score", f"{r2:.3f}", help="Closer to 1 generally indicates a better fit.")

st.caption(
    "These metrics are calculated on a test split. Ensure the split settings "
    "match your notebook to compare results fairly."
)

st.divider()

# -----------------------------
# Feature importance
# -----------------------------
st.subheader("🔎 Feature Importance")

try:
    fitted_rf = model.named_steps["model"]
    importances = fitted_rf.feature_importances_

    importance_df = pd.DataFrame({
        "Advertising Channel": features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df["Advertising Channel"], importance_df["Importance"])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Feature importance indicates how much each feature contributes to "
        "the model's tree-based decisions. It does not prove causation."
    )
except (AttributeError, KeyError):
    st.warning("Feature importance is unavailable for this saved model.")

st.divider()

# -----------------------------
# Interactive prediction
# -----------------------------
st.subheader("💰 Predict Sales")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Advertising", min_value=0.0, value=150.0, step=1.0)

with col2:
    radio = st.number_input("Radio Advertising", min_value=0.0, value=25.0, step=1.0)

with col3:
    newspaper = st.number_input(
        "Newspaper Advertising", min_value=0.0, value=20.0, step=1.0
    )

if st.button("🚀 Predict Sales", type="primary"):
    input_data = pd.DataFrame({
        "TV": [tv],
        "radio": [radio],
        "newspaper": [newspaper]
    })

    prediction = float(model.predict(input_data)[0])

    st.success(f"Predicted Sales: {prediction:.2f}")
    st.caption(
        "This is an estimate based on patterns in the training data. "
        "Actual sales may differ."
    )

    st.write("Your input budgets:")
    st.dataframe(input_data, hide_index=True, use_container_width=True)

st.divider()
st.caption("Built with Python, Pandas, Scikit-learn, Matplotlib, and Streamlit.")
