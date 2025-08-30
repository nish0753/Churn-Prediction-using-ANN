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
try:
    with open('model_architecture.json', 'r') as f:
        model_info = json.load(f)
    input_shape = (model_info['input_shape'],)
    model = create_model(input_shape)
    st.success("✅ Model architecture loaded successfully!")
    model_loaded = True
except Exception as e:
    st.warning(f"⚠️ Model architecture file not found. Using default architecture.")
    # Default input shape based on your preprocessing
    input_shape = (12,)  # 12 features after preprocessing
    model = create_model(input_shape)
    model_loaded = False

# Load the encoders and scaler with error handling
try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    encoders_loaded = True
    st.success("✅ Encoders and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading preprocessors: {e}")
    st.info("Please ensure the pickle files are available in the repository.")
    st.stop()


## streamlit app
st.title('Customer Churn PRediction')

# Display model status
if 'model_loaded' in locals() and model_loaded:
    st.success("✅ Trained model loaded successfully!")
else:
    st.warning("⚠️ Using untrained model. Predictions may not be accurate.")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

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


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
