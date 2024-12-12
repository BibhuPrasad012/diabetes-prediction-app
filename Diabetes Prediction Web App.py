# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:58:00 2024

@author: ACER
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open(r'D:\ML Stuff\diabetes-prediction-system\trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
        
def main():
    
    #Giving a title
    st.title("Diabetes Prediction Web App")
    
    #Getting the input data from user
    Pregnancies = st.text_input('Number of Pregnancies:')
    Glucose = st.text_input('Glucose Level:')
    BloodPressure = st.text_input('Blood Pressure:')
    SkinThickness = st.text_input('Skin Thickness:')
    Insulin = st.text_input('Insulin:')
    BMI = st.text_input('BMI:')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function:')
    Age = st.text_input('Age:')
    
    #Code for Prediction
    diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__== '__main__':
   main()