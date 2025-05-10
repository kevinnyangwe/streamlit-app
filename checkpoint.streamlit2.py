import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Streamlit app layout
st.title('Bank Account Prediction')

# Input fields for each feature
country = st.selectbox('Country', ['Country1', 'Country2', 'Country3', 'Country4'])
location_type = st.selectbox('Location Type', ['Urban', 'Rural'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
gender = st.selectbox('Gender', ['Male', 'Female'])
relationship_with_head = st.selectbox('Relationship with Head', ['Head', 'Spouse', 'Child', 'Other'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
education_level = st.selectbox('Education Level', ['Primary', 'Secondary', 'Tertiary'])
job_type = st.selectbox('Job Type', ['Employed', 'Self-employed', 'Unemployed'])

# Button to make prediction
if st.button('Predict'):
    # Map inputs to features for prediction (you may need to preprocess them just like in your training phase)
    input_data = {
        'country': country,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'gender': gender,
        'relationship_with_head': relationship_with_head,
        'marital_status': marital_status,
        'education_level': education_level,
        'job_type': job_type
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)

    # Display the result
    st.write(f"Prediction: {'Has Bank Account' if prediction == 1 else 'No Bank Account'}")
