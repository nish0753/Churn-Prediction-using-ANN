import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Function to create model with the same architecture as training
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model (using demo mode since model files aren't available in deployment)
st.title('🏦 Customer Churn Prediction')
st.markdown("AI-powered prediction system using Artificial Neural Networks")

# Info about demo mode
st.info("📝 **Demo Mode**: This version runs without the trained model weights for demonstration purposes.")

# Create model with expected input shape (12 features after preprocessing)
model = create_model((12,))

# Hard-coded geography categories (since encoder files aren't available)
geography_options = ['France', 'Germany', 'Spain']

# Create input form
st.subheader("📊 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    geography = st.selectbox('🌍 Geography', geography_options)
    gender = st.selectbox('👤 Gender', ['Male', 'Female'])
    age = st.slider('🎂 Age', 18, 92, 40)
    tenure = st.slider('⏰ Tenure (years)', 0, 10, 5)

with col2:
    st.markdown("**Financial Information**")
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('💵 Balance', min_value=0.0, value=60000.0)
    estimated_salary = st.number_input('� Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 2)

col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
with col4:
    is_active_member = st.selectbox('⚡ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button('🔮 Predict Churn', type='primary'):
    # Manual preprocessing (since encoder files aren't available)
    
    # Encode gender: Male=1, Female=0
    gender_encoded = 1 if gender == 'Male' else 0
    
    # One-hot encode geography
    geo_france = 1 if geography == 'France' else 0
    geo_germany = 1 if geography == 'Germany' else 0
    geo_spain = 1 if geography == 'Spain' else 0
    
    # Create input array with all features
    input_data = np.array([[
        credit_score, gender_encoded, age, tenure, balance, 
        num_of_products, has_cr_card, is_active_member, estimated_salary,
        geo_france, geo_germany, geo_spain
    ]])
    
    # Simple scaling (using typical values for demonstration)
    # In a real deployment, you'd use the saved scaler
    input_scaled = (input_data - np.array([[650, 0.5, 40, 5, 75000, 2, 0.7, 0.5, 100000, 0.33, 0.33, 0.33]])) / np.array([[100, 0.5, 15, 3, 50000, 1, 0.5, 0.5, 50000, 0.5, 0.5, 0.5]]))
    
    # Make prediction (will be random since model is untrained)
    prediction = model.predict(input_scaled, verbose=0)
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
            st.error("🚨 **High Risk**: Customer likely to churn!")
        else:
            st.success("✅ **Low Risk**: Customer likely to stay!")
    
    # Show customer profile
    st.markdown("---")
    st.subheader("📈 Customer Profile Summary")
    
    profile_data = {
        "Feature": ["Credit Score", "Age", "Tenure", "Balance", "Salary", "Products"],
        "Value": [credit_score, age, tenure, f"${balance:,.0f}", f"${estimated_salary:,.0f}", num_of_products]
    }
    
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, use_container_width=True)

# Disclaimer
st.markdown("---")
st.warning("⚠️ **Demo Mode**: This prediction is for demonstration only as it uses an untrained model. For accurate predictions, please run the full version locally with the trained model.")

# Footer
st.markdown("🤖 **AI-Powered Customer Churn Prediction** | Built with Streamlit & TensorFlow")
    pip install -r requirements.txt
    streamlit run app_fixed.py
    ```
    
    ### 📋 Required Files for Full Functionality:
    - `model_architecture.json` - Model configuration
    - `label_encoder_gender.pkl` - Gender encoder
    - `onehot_encoder_geo.pkl` - Geography encoder  
    - `scaler.pkl` - Feature scaler
    """)
    st.stop()

# Continue with full app if files exist
try:
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
    input_shape = (model_info['input_shape'],)
    model = create_model(input_shape)
    
    # Load the encoders and scaler
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

except Exception as e:
    st.error(f"❌ Error loading model components: {e}")
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
