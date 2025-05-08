import streamlit as st
import pandas as pd
import joblib

# Load model, encoders, metrics
model = joblib.load("career_model.pkl")
metrics = joblib.load("model_metrics.pkl")
all_metrics = joblib.load("all_model_metrics.pkl")
gender_encoder = joblib.load("Gender_encoder.pkl")
field_encoder = joblib.load("Field_of_Study_encoder.pkl")

st.title("Career Success Prediction Based on Education")

st.subheader("Model Evaluation Metrics (Decision Tree)")
for metric, value in metrics.items():
    st.write(f"{metric}: {value:.2f}")

st.subheader("Model Performance Comparison")
comparison_df = pd.DataFrame(all_metrics).T
st.dataframe(comparison_df.style.format("{:.2f}"))

st.subheader("Try a Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=60, value=18)
gender = st.selectbox("Gender", gender_encoder.classes_)
high_gpa = st.slider("High School GPA", 0.0, 4.0, 0.0)
sat = st.number_input("SAT Score", 400, 1600, 400)
ranking = st.number_input("University Ranking", 1, 1000, 1000)
uni_gpa = st.slider("University GPA", 0.0, 4.0, 0.0)
field = st.selectbox("Field of Study", field_encoder.classes_)
internships = st.slider("Internships Completed", 0, 10, 0)
projects = st.slider("Projects Completed", 0, 20, 0)
certs = st.slider("Certifications", 0, 10, 0)
soft = st.slider("Soft Skills Score", 0, 10, 0)
network = st.slider("Networking Score", 0, 10, 0)

# Encode input
input_data = pd.DataFrame([[
    age,
    gender_encoder.transform([gender])[0],
    high_gpa,
    sat,
    ranking,
    uni_gpa,
    field_encoder.transform([field])[0],
    internships,
    projects,
    certs,
    soft,
    network
]], columns=[
    'Age', 'Gender', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
    'University_GPA', 'Field_of_Study', 'Internships_Completed', 'Projects_Completed',
    'Certifications', 'Soft_Skills_Score', 'Networking_Score'
])

# Predict
if st.button("Predict Career Satisfaction"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Career Satisfaction (Decision Tree): {prediction}")
