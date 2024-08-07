import os
import numpy as np
import joblib as jb
import streamlit as st

catboost_model = jb.load('model_files/catboost_model.joblib')
ensemble_model = jb.load('model_files/ensemble_model.joblib')
lr_model = jb.load('model_files/lr_model.joblib')
svm_model = jb.load('model_files/svm_model.joblib')
mms_scaler = jb.load('model_files/mms.joblib')
ss_scaler = jb.load('model_files/ss.joblib')


st.title("Welcome to This-Ease(Hearts)")
st.subheader("An app to predict presence of any heart disease")

st.caption("Please enter the following details to predict the presence of heart disease")

# Number input ----------------------------************--------------------------
Age = int(st.number_input("Enter your age", min_value=1, max_value=100, value=22))
st.write('You have entered age:',Age)

Resting_BP = int(st.number_input("Enter your resting blood pressure", min_value=50, max_value=200, value=120))
st.write('You have entered resting blood pressure:',Resting_BP)

Cholesterol = int(st.number_input("Enter your cholesterol level", min_value=50, max_value=640, value=200))
st.write('You have entered cholesterol level:',Cholesterol)

MaxHR = int(st.number_input("Enter your maximum heart rate", min_value=50, max_value=240, value=160))
st.write('You have entered maximum heart rate:',MaxHR)

FastingBS = int(st.number_input("Enter your fasting blood sugar", min_value=0, max_value=1000, value=80)>120)
if FastingBS==1:
    st.write('You have fasting blood sugar > 120 mg/dl')
else:
    st.write('You have fasting blood sugar < 120 mg/dl')

Oldpeak = float(st.number_input("Enter your ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=0.0))
st.write('You have entered ST depression induced by exercise relative to rest:',Oldpeak)


# Radio type ----------------------------************--------------------------
Sex = int(st.radio("Select your gender", options=['M', 'F'])=='M')
st.write('You have selected gender:','M' if Sex==1 else 'F')

ChestPainType = int(st.radio("Select chest pain type", options=[0, 1, 2, 3]))
st.write('You have selected chest pain of type:',int(ChestPainType))

ExerciseAngina = int(st.radio("Do you have exercise induced angina?", options=[0, 1]))
if ExerciseAngina==1:
    st.write('You have exercise induced angina')
else:
    st.write('You do not have exercise induced angina')

RestingECG = st.selectbox("Select resting ECG", options=['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
if RestingECG=='Normal':
    RestingECG=0
elif RestingECG=='ST-T wave abnormality':
    RestingECG=1
else:
    RestingECG=2

ST_Slope = st.radio("Select ST slope", options=['Upsloping', 'Flat', 'Downsloping'])
st.write('You have selected ST slope:',ST_Slope)
if ST_Slope=='Upsloping':
    ST_Slope=0
elif ST_Slope=='Flat':
    ST_Slope=1
else:
    ST_Slope=2


inputs = [[Age, Sex, ChestPainType, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]]
standard_scaled_features = [0, 3, 5]
minmax_scaled_features = [7]

def scale_inputs(inputs, standard_scaler, minmax_scaler,
                  standard_scaled_features, minmax_scaled_features):
    scaled_inputs = np.array(inputs).copy()

    # if standard_scaled_features:
    #     scaled_features_standard = standard_scaler.transform(scaled_inputs[:, standard_scaled_features])
    #     scaled_inputs[:, standard_scaled_features] = scaled_features_standard

    # if minmax_scaled_features:
    #     scaled_features_minmax = minmax_scaler.transform(scaled_inputs[:, minmax_scaled_features])
    #     scaled_inputs[:, minmax_scaled_features] = scaled_features_minmax

    # return scaled_inputs
    scaled_inputs[0][0] = ss_scaler.transform([[inputs[0][0]]])
    scaled_inputs[0][5] = ss_scaler.transform([[inputs[0][5]]])
    scaled_inputs[0][3] = ss_scaler.transform([[inputs[0][5]]])
    scaled_inputs[0][7] = mms_scaler.transform([[inputs[0][7]]])
    return scaled_inputs


scaled_inputs = scale_inputs(inputs, ss_scaler, mms_scaler, standard_scaled_features, minmax_scaled_features)
model_type = st.selectbox("Models", options=['Logistic Regression', 'Catboost', 'SVM', 'Ensemble'])

st.text("Please click the button below to predict the presence of heart disease")
if model_type=='Logistic Regression':
    model = lr_model
elif model_type=='Catboost':
    model = catboost_model
elif model_type=='SVM':
    model = svm_model
else:
    model = ensemble_model

if st.button("Predict"):
    prediction = model.predict(scaled_inputs)[0]
    if prediction==1:
        st.write("You may have heart disease! Please consult a doctor immediately!")
    else:
        st.write("No need to worry and raise a red flag! You do not have heart disease")