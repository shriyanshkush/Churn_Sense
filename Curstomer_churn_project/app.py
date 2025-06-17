import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import time

# Gemini
import google.generativeai as genai

# Set API key
genai.configure(api_key="AIzaSyCMr48MKHcNkx1HCToTRDaLqwSHn34TMho")

# Load Gemini model
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# Load Pickle Files
def load_pickle_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Define paths
MODEL_PATH = os.path.join(os.getcwd(), "churn_model_final.pkl")
ENCODERS_PATH = os.path.join(os.getcwd(), "encoders.pkl")

# Load model and encoders
model = load_pickle_file(MODEL_PATH)
encoders = load_pickle_file(ENCODERS_PATH)

if isinstance(model, dict):
    raise TypeError("Loaded 'churn_model.pkl' is a dictionary, not a trained model!")

# 🎨 Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# 📌 Sidebar for Inputs
st.sidebar.header("🔍 Enter Customer Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"], index=1)
senior_citizen = st.sidebar.radio("Senior Citizen", [0, 1])
partner = st.sidebar.radio("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.radio("Has Dependents?", ["Yes", "No"])
tenure = st.sidebar.slider("📆 Tenure (Months)", min_value=0, max_value=100, value=10)
phone_service = st.sidebar.radio("📞 Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.radio("🔐 Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.radio("💾 Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.radio("🛡 Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.radio("🛠 Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.radio("📺 Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.radio("🎬 Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("📝 Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("💵 Monthly Charges ($)", min_value=0.0, max_value=500.0, step=0.1, value=50.0)
total_charges = st.sidebar.number_input("💰 Total Charges ($)", min_value=0.0, max_value=10000.0, step=0.1, value=500.0)

# 📥 Prepare input
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_data_df = pd.DataFrame([input_data])

# Encode categorical variables
for column in input_data_df.columns:
    if column in encoders:
        input_data_df[column] = encoders[column].transform(input_data_df[column])

# 🔘 Predict button
if st.sidebar.button("🔮 Predict Churn"):
    with st.spinner("🔄 Analyzing Customer Data..."):
        time.sleep(2)
    
    prediction = model.predict(input_data_df)[0]

    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(input_data_df)[0]
        churn_prob = round(pred_prob[1] * 100, 2)
        no_churn_prob = round(pred_prob[0] * 100, 2)

        # Save to session state
        st.session_state['prediction'] = prediction
        st.session_state['churn_prob'] = churn_prob
        st.session_state['no_churn_prob'] = no_churn_prob

# 📊 Show prediction if available
if 'prediction' in st.session_state:
    st.subheader("📊 Prediction Result")
    st.markdown("### 🔥 Churn Probability Breakdown")
    st.progress(int(st.session_state['churn_prob']))

    col1, col2 = st.columns(2)
    col1.metric("🟢 No Churn Probability", f"{st.session_state['no_churn_prob']}%")
    col2.metric("🔴 Churn Probability", f"{st.session_state['churn_prob']}%")

    if st.session_state['prediction'] >= 0.60:
        st.error("❌ **Churn Prediction: Yes**")
    else:
        st.success("✅ **Churn Prediction: No**")

# --- Gemini Section ---
st.markdown("---")
st.subheader("🧠 Smart Plan & Coupon Generator")

with st.form("gemini_prompt_form"):
    user_prompt = st.text_area("✍️ Describe what you need (e.g., 'Generate a discount coupon strategy for churned customers')")

    submitted = st.form_submit_button("Generate")

    if submitted and user_prompt.strip():
        with st.spinner("🧠 Thinking..."):
            try:
                response = gemini_model.generate_content(user_prompt)
                st.success("✅ Here's a generated strategy based on your input:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"❌ Failed to get response: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<h5 style='text-align: center; color: grey;'>🔍 Powered by Machine Learning</h5>",
    unsafe_allow_html=True
)
