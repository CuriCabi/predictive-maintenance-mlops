
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="tprasuna/predictive-maintenance-randomforest",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

st.title("Predictive Maintenance System")

st.write("Enter engine sensor values:")

engine_rpm = st.number_input("Engine RPM", value=700)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=6.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=77.0)
coolant_temp = st.number_input("Coolant Temperature", value=78.0)

if st.button("Predict Engine Condition"):

    input_df = pd.DataFrame({
        "Engine rpm": [engine_rpm],
        "Lub oil pressure": [lub_oil_pressure],
        "Fuel pressure": [fuel_pressure],
        "Coolant pressure": [coolant_pressure],
        "lub oil temp": [lub_oil_temp],
        "Coolant temp": [coolant_temp]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Maintenance Required")
    else:
        st.success("Engine Operating Normally")
