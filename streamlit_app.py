import streamlit as st
import pickle
import numpy as np

# Load trained model, scaler, and encoders
model_filename = "trained_modell.pkl"
with open(model_filename, "rb") as file:
    model, scaler, label_encoders = pickle.load(file)

st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

# User input fields
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
age = st.slider("Age", 0, 100, 25)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox("Family History of Overweight", label_encoders["family_history_with_overweight"].classes_)
favc = st.selectbox("High Calorie Food Consumption", label_encoders["FAVC"].classes_)
fcvc = st.slider("Vegetable Consumption Frequency", 1.0, 3.0, 2.0)
ncp = st.slider("Number of Meals per Day", 1.0, 4.0, 3.0)
caec = st.selectbox("Consumption of Food Between Meals", label_encoders["CAEC"].classes_)
smoke = st.selectbox("Do you smoke?", label_encoders["SMOKE"].classes_)
ch2o = st.slider("Water Consumption (liters per day)", 1.0, 3.0, 2.0)
scc = st.selectbox("Calories Monitoring?", label_encoders["SCC"].classes_)
faf = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0)
tue = st.slider("Time Using Technology (hours)", 0.0, 2.0, 1.0)
calc = st.selectbox("Alcohol Consumption", label_encoders["CALC"].classes_)
mtrans = st.selectbox("Mode of Transportation", label_encoders["MTRANS"].classes_)

# Convert inputs to numerical format
input_data = np.array([
    label_encoders["Gender"].transform([gender])[0],
    age, height, weight,
    label_encoders["family_history_with_overweight"].transform([family_history])[0],
    label_encoders["FAVC"].transform([favc])[0],
    fcvc, ncp,
    label_encoders["CAEC"].transform([caec])[0],
    label_encoders["SMOKE"].transform([smoke])[0],
    ch2o, label_encoders["SCC"].transform([scc])[0],
    faf, tue,
    label_encoders["CALC"].transform([calc])[0],
    label_encoders["MTRANS"].transform([mtrans])[0]
]).reshape(1, -1)

# Normalize numerical values
input_data[:, [1, 2, 3, 6, 7, 10, 12, 13]] = scaler.transform(input_data[:, [1, 2, 3, 6, 7, 10, 12, 13]])

# Predict obesity level
prediction = model.predict(input_data)
predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]

st.success(f"Predicted Obesity Level: **{predicted_class}**")
