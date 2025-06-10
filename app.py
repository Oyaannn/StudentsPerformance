import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prediksi Kelulusan Siswa", page_icon="🎓")

st.title("🎓 Prediksi Kelulusan Siswa dengan KNN")

# Form input
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
race = st.selectbox("Ras/Etnis", ["Grup A", "Grup B", "Grup C", "Grup D", "Grup E"])
education = st.selectbox("Pendidikan Orang Tua", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Tipe Makan Siang", ["standart", "free/reduced"])
prep = st.selectbox("Kursus Persiapan Ujian", ["Tidak", "Ya"])
math = st.slider("Nilai Matematika", 0, 100, 50)
reading = st.slider("Nilai Membaca", 0, 100, 50)
writing = st.slider("Nilai Menulis", 0, 100, 50)

if st.button("Prediksi Kelulusan"):
    # Konversi input ke numerik (berdasarkan urutan LabelEncoder saat training)
    gender_num = 0 if gender == "Laki-laki" else 1
    race_dict = {"Grup A": 0, "Grup B": 1, "Grup C": 2, "Grup D": 3, "Grup E": 4}
    education_dict = {
        "some high school": 5, "high school": 3, "some college": 4,
        "associate's degree": 0, "bachelor's degree": 1, "master's degree": 2
    }
    lunch_num = 1 if lunch == "standart" else 0
    prep_num = 1 if prep == "Ya" else 0

    data = np.array([[gender_num, race_dict[race], education_dict[education],
                      lunch_num, prep_num, math, reading, writing]])
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    st.success(f"✅ Prediksi: **{prediction}**")
