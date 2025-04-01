import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pickle import load

model = load(open('models/mhr_rforest_model.pkl', 'rb'))
class_dic = {'0': 'high risk', '1': 'low risk', '2':' mid risk'}

st.markdown('<style>description{color:white;}</style>', unsafe_allow_html=True)

st.markdown("<description> Data has been collected from different hospitals, " +
"community clinics, maternal health cares from the rural areas" +
"of Bangladesh through the IoT based risk monitoring system. </description>", unsafe_allow_html=True)

st.title("Maternal Health Risk - Model prediction")

age = st.slider("Choose your age: ", min_value=10, max_value=66, value=25, step=1)
systolicbp = st.slider("Choose your systolicbp: ", min_value=70, max_value=160, value=120, step=1)
diastolicbp = st.slider("Choose your diastolicbp: ", min_value=49, max_value=100, value=80, step=1)
bs = st.slider("Choose your bs: ", min_value=6.0, max_value=19.0, value=7.5, step=0.1)
bodytemp = st.slider("Choose your bodytemp: ", min_value=98, max_value=103, value=99, step=1)
heartrate = st.slider("Choose your heartrate: ", min_value=40, max_value=90, value=76, step=1)


if st.button("Realizar predicción"):
    prediction = str(model.predict([[age, systolicbp, diastolicbp, bs, bodytemp, heartrate]])[0])
    pred_class = class_dic[prediction]
    st.write("Prediction:", pred_class)