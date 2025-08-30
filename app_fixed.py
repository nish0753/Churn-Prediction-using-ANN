import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import json

# Function to create model with the same architecture as training
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load model architecture info
with open('model_architecture.json', 'r') as f:
    model_info = json.load(f)

# Create model with correct architecture
input_shape = (model_info['input_shape'],)
model = create_model(input_shape)

# Try to load weights from the original model
try:
    # Load the original model and transfer weights
    original_model = tf.keras.models.load_model('model.h5')
    model.set_weights(original_model.get_weights())
    st.success("✅ Model weights loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load original model weights: {e}")
    st.info("Using untrained model for demonstration. Please retrain the model.")

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('🏦 Customer Churn Prediction')
st.markdown("Predict whether a customer will leave the bank based on their profile.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Demographics")
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', ['Male', 'Female'])
    age = st.slider('🎂 Age', 18, 92, 40)
    tenure = st.slider('⏰ Tenure (years)', 0, 10, 5)

with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('💵 Balance', min_value=0.0, value=60000.0)
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 2)

st.subheader("🏦 Account Details")
col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
with col4:
    is_active_member = st.selectbox('⚡ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button('🔮 Predict Churn', type='primary'):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [1 if gender == 'Male' else 0],  # Manual encoding: Male=1, Female=0
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]])
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled, verbose=0)
        prediction_proba = prediction[0][0]

        # Display results
        st.markdown("---")
        
        col_result1, col_result2 = st.columns([1, 1])
        
        with col_result1:
            st.metric(
                label="🎯 Churn Probability", 
                value=f"{prediction_proba:.1%}"
            )
        
        with col_result2:
            if prediction_proba > 0.5:
                st.error("🚨 **High Risk**: Customer is likely to churn!")
                st.markdown("**Recommendations:**")
                st.markdown("- Offer retention incentives")
                st.markdown("- Improve customer service")
                st.markdown("- Provide personalized offers")
            else:
                st.success("✅ **Low Risk**: Customer is likely to stay!")
                st.markdown("**Recommendations:**")
                st.markdown("- Continue current service level")
                st.markdown("- Consider upselling opportunities")
                st.markdown("- Maintain regular engagement")

        # Additional insights
        st.markdown("---")
        st.subheader("📈 Customer Profile Summary")
        
        profile_data = {
            "Feature": ["Credit Score", "Age", "Tenure", "Balance", "Salary", "Products"],
            "Value": [credit_score, age, tenure, f"${balance:,.0f}", f"${estimated_salary:,.0f}", num_of_products],
            "Category": [
                "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair",
                "Senior" if age >= 60 else "Middle-aged" if age >= 35 else "Young",
                "Long-term" if tenure >= 7 else "Medium-term" if tenure >= 3 else "New",
                "High" if balance >= 100000 else "Medium" if balance >= 20000 else "Low",
                "High" if estimated_salary >= 80000 else "Medium" if estimated_salary >= 40000 else "Low",
                "Multiple" if num_of_products >= 3 else "Few" if num_of_products == 2 else "Single"
            ]
        }
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("🤖 **AI-Powered Customer Churn Prediction** | Built with Streamlit & TensorFlow")
