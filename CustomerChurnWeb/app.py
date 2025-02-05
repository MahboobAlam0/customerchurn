import streamlit as st
import pandas as pd
import numpy as np
import pickle
import numpy as np

def predict_churn(features):
    # Convert categorical features to numeric
    gender_encoded = 1 if features[6] == "Male" else 0  # Assuming Male = 1, Female = 0
    phone_service_encoded = 1 if features[7] == "Yes" else 0  # Yes = 1, No = 0
    multipleLines_encoded = 1 if features[8] == "Yes" else 0  # Yes = 1, No = 0
    
    # Replace categorical values in the feature list
    numeric_features = [
        features[0],  # tenure
        features[1],  # monthly_charges
        features[2],  # total_charges
        features[3],  # contract_type_encoded
        features[4],  # payment_method_encoded
        features[5],  # internet_service_encoded
        gender_encoded,
        phone_service_encoded, 
        multipleLines_encoded
    ]

    # Convert to NumPy array and reshape for model input
    features_array = np.array(numeric_features, dtype=float).reshape(1, -1)

    # Get churn probability
    prediction = model.predict_proba(features_array)[:, 1]
    return prediction[0]

# Function to load and apply CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="centered"
)

# Load CSS
load_css('style.css')

# Custom Header
st.markdown("""
<div class="header">
    <h1>CHURN <span>PREDICTOR</span></h1>
</div>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
This app predicts the probability of customer churn based on various factors.
Enter the customer information below to get a prediction.
""")


# Create input form
st.subheader("Customer Information")

with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, value=0)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=0.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.0)
        gender = st.selectbox("Select Gender", ["Male", "Female"])  
    
    with col2:
        contract_type = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multipleLines = st.selectbox(
            "Multiple Lines", ["Yes", "No"]
        )
    
    submit_button = st.form_submit_button("Predict Churn Probability")

# Make prediction when form is submitted
if submit_button:
    # Convert categorical variables to numeric
    contract_type_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract_type]
    payment_method_encoded = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[payment_method]
    internet_service_encoded = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
    
    # Create feature vector
    features = [
        tenure,
        monthly_charges,
        total_charges,
        contract_type_encoded,
        payment_method_encoded,
        internet_service_encoded, 
        gender, 
        phone_service,
        multipleLines
    ]
    
    # Get prediction
    churn_probability = predict_churn(features)
    
    # Display results with nice formatting
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # Create columns for visualization
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Display probability as a percentage
        churn_percentage = churn_probability * 100
        
        # Color coding based on risk level
        if churn_percentage < 30:
            color = "green"
            risk_level = "Low Risk"
        elif churn_percentage < 60:
            color = "orange"
            risk_level = "Medium Risk"
        else:
            color = "red"
            risk_level = "High Risk"
        
        st.markdown(f"""
        <div style='text-align: center;'>
            <h3 style='color: {color};'>{risk_level}</h3>
            <h2 style='color: {color};'>{churn_percentage:.1f}%</h2>
            <p>Probability of Churn</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### Key Insights")
        if contract_type == "Month-to-month":
            st.write("- Month-to-month contracts have higher churn risk")
        if tenure < 12:
            st.write("- New customers need extra attention")
        if monthly_charges > 80:
            st.write("- High monthly charges may increase churn risk")

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This is a demonstration model using a Random Forest Classifier. In a production environment, you would use:
    - A properly trained model with real historical data
    - More features for better prediction accuracy
    - Regular model retraining and validation
    - Proper model versioning and monitoring
    """)
