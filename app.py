import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load models
@st.cache_resource
def load_pickle_model(path):
   with open(path, 'rb') as file:
       return pickle.load(file)

@st.cache_resource
def load_keras_model(path):
   return load_model(path)

logistic_regression_model = load_pickle_model('Binary/Logistic_Regression.pkl')
random_forest_model = load_pickle_model('Binary/Random_Forest.pkl')
cnn_model = load_keras_model('MultiClass/cnn_model.h5')
ann_model = load_keras_model('MultiClass/ann_model.h5')
cnn_explainable_model = load_keras_model('Explainable/cnn_model_explainable.h5')
ann_explainable_model = load_keras_model('Explainable/ann_model_explainable.h5')

# User interface
st.title('Model Testing Web App')

model_type = st.selectbox('Select Model Type', ('Binary', 'MultiClass', 'Explainable'))
model_name = st.selectbox('Select Model', {
   'Binary': ['Logistic Regression', 'Random Forest'],
   'MultiClass': ['CNN', 'ANN'],
   'Explainable': ['CNN Explainable', 'ANN Explainable']
}[model_type])

uploaded_file = st.file_uploader('Upload CSV for prediction', type='csv')

if uploaded_file:
   data = pd.read_csv(uploaded_file)
   st.write('Uploaded Data:')
   st.write(data.head())

   if st.button('Predict'):
       if model_type == 'Binary':
           if model_name == 'Logistic Regression':
               model = logistic_regression_model
           else:
               model = random_forest_model
       elif model_type == 'MultiClass':
           if model_name == 'CNN':
               model = cnn_model
           else:
               model = ann_model
       else:
           if model_name == 'CNN Explainable':
               model = cnn_explainable_model
           else:
               model = ann_explainable_model

       # Make predictions
       predictions = model.predict(data)
       st.write('Predictions:')
       st.write(predictions)
