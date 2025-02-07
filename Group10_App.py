# -*- coding: utf-8 -*-


import os
import numpy as np
import pickle
import streamlit as st

# Define paths
model_path = os.path.join(os.path.dirname(__file__), "diagnosis_trained_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load model and scaler
loaded_model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

#creating function
def diabetes_prediction(input_data):
    # Convert to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Standardize the input using the saved scaler
    standardized_data = scaler.transform(input_data_as_numpy_array)

    # Make prediction
    prediction = loaded_model.predict(standardized_data)

    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'  

    
    
def main():
    
    st.title('Diabetes Prediction')
    
  
    
    Pregnancies=st.slider('Number of Pregnancies',min_value=0,max_value=20,value=4,step=1)
    Glucose=st.slider('Glucose level',min_value=0,max_value=199,value=60,step=1)
    BloodPressure=st.slider('BloodPressure level in mm Hg',min_value=0,max_value=140,value=80,step=1)
    BMI=st.slider('BMI value',min_value=0.0,max_value=70.0,value=33.3,step=0.01)
    DiabetesPedigreeFunction=st.slider('Diabetes Pedigree Function value',min_value=0.000, step=0.001, max_value=3.0, value=0.045, format="%3f")
    Age=st.slider('Age of a person',min_value=10,max_value=100,value=21,step=1)
   
    
   
    diagnosis = ''
    if st.button('Dibetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)    
    


if __name__ == '__main__':
     main()    
    
    
