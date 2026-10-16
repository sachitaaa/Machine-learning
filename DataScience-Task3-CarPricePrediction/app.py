
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()
feature_names = list(model.feature_names_in_)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #101827, #203b63);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .result-note {
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🚘 CarPrice AI</h1>
    <p>Estimate a used car's selling price with machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "A machine learning project using a trained Decision Tree Regressor."
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("About this app")
    st.write(
        "Enter the car's specifications to receive an estimated selling price."
    )
    st.divider()
    st.write("**Model:** Decision Tree Regressor")
    st.write("**Input features:** 55")
    st.write("**Task:** Regression")
    st.divider()
    st.caption(
        "Predictions depend on the training data and may not reflect "
        "current market prices."
    )

# ---------------- INPUT FORM ----------------
st.markdown(
    '<div class="section-title">Car specifications</div>',
    unsafe_allow_html=True
)
st.write("Provide the details below.")

with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input(
            "Current/present price (lakhs)",
            min_value=0.1,
            max_value=1000.0,
            value=8.0,
            step=0.5
        )

        kms_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=1_000_000,
            value=30000,
            step=1000
        )

        car_age = st.number_input(
            "Car age (years)",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )

        owner = st.selectbox(
            "Previous owners",
            options=[0, 1, 2, 3]
        )

    with col2:
        fuel_type = st.selectbox(
            "Fuel type",
            options=["Petrol", "Diesel", "Cng"]
        )

        seller_type = st.selectbox(
            "Seller type",
            options=["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "Transmission",
            options=["Manual", "Automatic"]
        )

        brand_options = [
            name.removeprefix("Brand_")
            for name in feature_names
            if name.startswith("Brand_")
        ]

        brand = st.selectbox(
            "Car category (from dataset)",
            options=brand_options,
            index=brand_options.index("i20")
            if "i20" in brand_options else 0
        )

    submitted = st.form_submit_button(
        "Estimate selling price",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    input_data.loc[0, "Present_Price"] = present_price
    input_data.loc[0, "Kms_Driven"] = kms_driven
    input_data.loc[0, "Car_Age"] = car_age
    input_data.loc[0, "Owner"] = owner

    input_data.loc[0, f"Fuel_Type_{fuel_type}"] = 1
    input_data.loc[0, f"Seller_Type_{seller_type}"] = 1
    input_data.loc[0, f"Transmission_{transmission}"] = 1
    input_data.loc[0, f"Brand_{brand}"] = 1

    predicted_price = float(model.predict(input_data)[0])

    st.divider()
    st.markdown("### Your estimated car price")

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        st.metric(
            label="Predicted selling price",
            value=f"{predicted_price:.2f} lakhs"
        )
        st.caption(
            "This is a model-generated estimate, not a guaranteed selling price."
        )

    with result_col2:
        st.metric(
            label="Car age",
            value=f"{car_age} years"
        )
        st.metric(
            label="Distance driven",
            value=f"{kms_driven:,} km"
        )

    with st.expander("View submitted specifications"):
        st.write({
            "Present price (lakhs)": present_price,
            "Kilometers driven": kms_driven,
            "Car age (years)": car_age,
            "Previous owners": owner,
            "Fuel type": fuel_type,
            "Seller type": seller_type,
            "Transmission": transmission,
            "Car category": brand
        })

st.divider()
st.caption("CarPrice AI | Machine Learning Portfolio Project")
