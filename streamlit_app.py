import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Fungsi untuk memuat model
def load_model(filename):
    with open(filename, "rb") as file:
        model, scaler, label_encoders = pickle.load(file)
    return model, scaler, label_encoders

# Fungsi prediksi
def predict_with_model(model, scaler, label_encoders, user_input):
    # Konversi input kategori ke numerik
    user_input[0] = label_encoders["Gender"].transform([user_input[0]])[0]
    user_input[4] = label_encoders["family_history_with_overweight"].transform([user_input[4]])[0]
    user_input[5] = label_encoders["FAVC"].transform([user_input[5]])[0]
    user_input[8] = label_encoders["CAEC"].transform([user_input[8]])[0]
    user_input[9] = label_encoders["SMOKE"].transform([user_input[9]])[0]
    user_input[11] = label_encoders["SCC"].transform([user_input[11]])[0]
    user_input[14] = label_encoders["CALC"].transform([user_input[14]])[0]
    user_input[15] = label_encoders["MTRANS"].transform([user_input[15]])[0]

    # Skalakan fitur numerik
    numerical_indices = [1, 2, 3, 6, 7, 10, 12, 13]
    user_input[numerical_indices] = scaler.transform([user_input[numerical_indices]])[0]

    # Konversi ke array dan prediksi
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]
    
    return predicted_class

def main():
    st.title('Obesity Prediction App')
    st.info('This app uses Machine Learning to predict obesity levels.')

    # **Menampilkan Raw Data di Awal**
    try:
        dataset_path = "/mnt/data/ObesityDataSet_raw_and_data_sinthetic.csv"
        df = pd.read_csv(dataset_path)
        
        with st.expander("📊 Data", expanded=True):
            st.write("This is a raw data")
            st.dataframe(df.head(10))  # Menampilkan 10 data pertama
        
    except FileNotFoundError:
        st.error("Dataset not found. Please upload the dataset.")

    # **User Input**
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', min_value=0, max_value=100, value=25)
    height = st.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0)
    family_history = st.selectbox('Family History of Overweight', ['yes', 'no'])
    favc = st.selectbox('Frequent Consumption of High-Calorie Food', ['yes', 'no'])
    fcvc = st.slider('Vegetable Consumption Frequency', min_value=1.0, max_value=3.0, value=2.0)
    ncp = st.slider('Number of Meals per Day', min_value=1.0, max_value=4.0, value=3.0)
    caec = st.selectbox('Consumption of Food Between Meals', ['no', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox('Do you smoke?', ['yes', 'no'])
    ch2o = st.slider('Water Consumption (liters per day)', min_value=1.0, max_value=3.0, value=2.0)
    scc = st.selectbox('Calories Monitoring?', ['yes', 'no'])
    faf = st.slider('Physical Activity Frequency', min_value=0.0, max_value=3.0, value=1.0)
    tue = st.slider('Time Using Technology (hours)', min_value=0.0, max_value=2.0, value=1.0)
    calc = st.selectbox('Alcohol Consumption', ['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox('Mode of Transportation', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

    # **Kumpulkan input dalam list**
    user_input = [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]

    # **Load model dan lakukan prediksi**
    model_filename = "obesity_model.pkl"
    try:
        model, scaler, label_encoders = load_model(model_filename)
        prediction = predict_with_model(model, scaler, label_encoders, user_input)
        st.success(f'The predicted obesity level is: **{prediction}**')
    except FileNotFoundError:
        st.error("Model file not found. Please upload the trained model.")

if __name__ == "__main__":
    main()


# # Scatter plot untuk visualisasi hubungan antara tinggi badan, berat badan, dan obesitas
# with st.expander("📊 Data Visualization", expanded=True):
#     st.subheader("Data Visualization")
    
#     fig = px.scatter(
#         df, 
#         x="Height", 
#         y="Weight", 
#         color="NObeyesdad",  # Warna berdasarkan kategori obesitas
#         title="Height vs Weight Distribution by Obesity Level"
#     )
    
    st.plotly_chart(fig)

