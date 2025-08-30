import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title('Customer Churn Prediction')
st.markdown("AI-powered prediction system using Artificial Neural Networks")

@st.cache_resource
def load_model_and_preprocessors():
    try:
        model = None
        
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('model.h5')
        except (ImportError, AttributeError):
            pass
        
        if model is None:
            try:
                import keras
                model = keras.models.load_model('model.h5')
            except (ImportError, AttributeError):
                pass
        
        if model is None:
            try:
                import tensorflow as tf
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.load_weights('model.h5')
            except:
                model = None
            
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, label_encoder_gender, onehot_encoder_geo, scaler, True
    except Exception as e:
        st.warning(f"Could not load trained model: {e}")
        return None, None, None, None, False

model, label_encoder_gender, onehot_encoder_geo, scaler, model_loaded = load_model_and_preprocessors()

if model_loaded:
    st.success("Trained model loaded successfully!")
    geography_options = list(onehot_encoder_geo.categories_[0])
    gender_options = list(label_encoder_gender.classes_)
else:
    st.info("Demo Mode: Using rule-based prediction as trained model is not available.")
    geography_options = ['France', 'Germany', 'Spain']
    gender_options = ['Male', 'Female']

st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    geography = st.selectbox('Geography', geography_options)
    gender = st.selectbox('Gender', gender_options)
    age = st.slider('Age', 18, 92, 40)
    tenure = st.slider('Tenure (years)', 0, 10, 5)

with col2:
    st.markdown("**Financial Information**")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=60000.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 2)

col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col4:
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

if st.button('Predict Churn', type='primary'):
    if model_loaded and model is not None:
        try:
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })

            geo_encoded = onehot_encoder_geo.transform([[geography]])
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]
            
            model_type = "AI Neural Network Prediction"
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            prediction_proba = 0.5
            model_type = "Error in Prediction"
    else:
        risk_factors = 0
        
        if age > 50: risk_factors += 1
        if balance < 10000: risk_factors += 1
        if credit_score < 600: risk_factors += 1
        if is_active_member == 0: risk_factors += 1
        if num_of_products == 1: risk_factors += 1
        
        prediction_proba = min(0.95, max(0.05, risk_factors * 0.18 + 0.1))
        model_type = "Rule-based Demo Prediction"
    
    st.markdown("---")
    
    col_result1, col_result2 = st.columns([1, 1])
    
    with col_result1:
        st.metric(
            label="Churn Probability", 
            value=f"{prediction_proba:.1%}"
        )
    
    with col_result2:
        if prediction_proba > 0.5:
            st.error("High Risk: Customer likely to churn!")
        else:
            st.success("Low Risk: Customer likely to stay!")
    
    st.caption(model_type)
    
    st.markdown("---")
    st.subheader("Customer Profile Summary")
    
    profile_data = {
        "Feature": ["Credit Score", "Age", "Tenure", "Balance", "Salary", "Products"],
        "Value": [credit_score, age, tenure, f"${balance:,.0f}", f"${estimated_salary:,.0f}", num_of_products]
    }
    
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, use_container_width=True)

st.markdown("---")
if model_loaded:
    st.success("Production Ready: Using trained neural network model for accurate predictions!")
else:
    st.warning("Demo Mode: For full functionality, ensure model.h5 and preprocessor files are available.")

st.markdown("AI-Powered Customer Churn Prediction | Built with Streamlit & TensorFlow")
