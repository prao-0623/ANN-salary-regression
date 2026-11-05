import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from pathlib import Path

base_dir = Path(__file__).resolve().parent

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(12,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.load_weights(base_dir / 'regression_model.h5')

#load encoders and scalers
with open(base_dir / 'label_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open(base_dir / 'label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open(base_dir / 'scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app
st.title('Customer Salary prediction')

if 'EstimatedSalary' in scaler.feature_names_in_:
    st.error('The loaded scaler belongs to the churn model. Save the salary model scaler before running this app.')
    st.stop()

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

geo_encode = label_encoder_geo.transform(pd.DataFrame({'Geography': [geography]})).toarray()
geo_encode_df = pd.DataFrame(geo_encode, columns=label_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encode_df], axis=1)
input_data = input_data.loc[:, scaler.feature_names_in_]
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled, verbose=0)
predicted_salary = prediction[0][0]
st.write(f'Predicted Salary: {predicted_salary:.2f}')