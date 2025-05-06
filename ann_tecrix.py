import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 

## Load the trained model 
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file) 

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file) 

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)  

# Function to preprocess the input data
st.title("Fintech Bank Customer Exitting Prediction")
 
# User inputs 

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

gender = st.selectbox('Gender', label_encoder_gender.classes_)

age = st.slider('Age', 18,100) 

balance = st.number_input('Balance')

CreditScore = st.number_input('CreditScore')

EstimatedSalary = st.number_input('EstimatedSalary')

tenure = st.slider('Tenure', 0,10)

NumOfProducts = st.slider('NumofProducts', 1, 5)

HasCrCard = st.selectbox('HasCrCard', [0,1,2,3])

IsActiveMember = st.selectbox('IsActiveMember', [0,])

#Prepare the input data for prediction

input_data = pd.DataFrame({
    'CreditScore' : [CreditScore],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [NumOfProducts],
    'HasCrCard' : [HasCrCard],
    'IsActiveMember' : [IsActiveMember],
    'EstimatedSalary' : [EstimatedSalary],

})


geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography'])) 


#Combine one_hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Scale the input data

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_proba = prediction [0] [0]

st.write(f"Churn Probability: {prediction_proba:.5}")

if prediction_proba > 0.10:
    st.write("The Customer will leave the bank")
else: 
    st.write("The Customer will not leave the bank")

