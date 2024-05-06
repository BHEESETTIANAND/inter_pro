import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

model=joblib.load("ranfor.pkl")
city_enc=pickle.load(open("city_encoder.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
trip=pickle.load(open("trip_encoder.pkl","rb"))
veh=pickle.load(open("veh_encoder.pkl","rb"))

st.title("Ticket Cancellation prediction")

f1=st.number_input("enter you reservation status")
f2=st.radio("is your gender is male",["True","False"])
f3=st.number_input("enter your ticket price")
f4=st.number_input("enter your coupon discount if eliglble or else enter 0")
f5=st.number_input("if your trip is domestic then enter 1 or else enter 0")
f6=st.radio("select your trip reason",['Work', 'Int'])
f7=st.radio("select the vehicle you are going to travel",['Plane', 'Bus', 'Train', 'InternationalPlane'])
f8=st.radio("select the city from where your are travelling",['گرگان', 'مشهد', 'شیراز', 'تبریز', 'تهران', 'بروجرد'])
f9=st.radio("select the city to which you are travelling",['گرگان', 'مشهد', 'شیراز', 'تبریز', 'تهران', 'بروجرد'])
f10=st.number_input("no.of.days staying")
f11=st.number_input("total count of travellers")
f12=st.number_input("total no.of.tickets booked")

if f2=="True":
    f2=1
else:
    f2=0

f6=trip.transform([f6])[0]
f7=veh.transform([f7])[0]
f8=city_enc.transform([f8])[0]
f9=city_enc.transform([f9])[0]

features=np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
features=scaler.transform([features])

prediction = model.predict(features.reshape(1, -1))

if(st.button("Predict")):
    if prediction[0] == 1:
        st.write("this customer can cancell the tickets")
    else:
        st.write("the customer will not cancell the tickets")