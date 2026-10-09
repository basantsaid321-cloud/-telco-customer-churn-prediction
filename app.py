import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "churn_pipeline.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Telco Churn")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Customer Prediction"]
)

# -----------------------------
# Home Page
# -----------------------------

if page == "Home":

    st.title("📊 Telco Customer Churn Prediction")

    st.markdown(
        """
        Welcome to the Telco Customer Churn Prediction App.

        This application uses a trained Machine Learning model
        to estimate whether a customer may leave the company.

        Navigate to **Customer Prediction** to enter customer
        information and receive a prediction.
        """
    )

    st.info(
        "The prediction is an estimate, not a guarantee "
        "of future customer behavior."
    )

# -----------------------------
# Customer Prediction Page
# -----------------------------

elif page == "Customer Prediction":

    st.title("🔍 Customer Churn Prediction")

    st.write(
        "Enter customer details below to estimate churn risk."
    )

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

        with col2:

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn"
        )

    if submitted:

        customer_data = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )

        try:

            prediction = int(
                model.predict(customer_data)[0]
            )

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "The model predicts that the customer "
                    "may churn."
                )

            else:

                st.success(
                    "The model predicts that the customer "
                    "may stay."
                )

            if hasattr(model, "predict_proba"):

                probability = float(
                    model.predict_proba(
                        customer_data
                    )[0, 1]
                )

                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.1%}"
                )

        except Exception as error:

            st.error(
                "Prediction failed. Check the model input "
                f"columns and data types. Details: {error}"
            )