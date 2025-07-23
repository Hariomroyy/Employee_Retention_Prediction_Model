# Employee Retention Predictor App using Streamlit

import streamlit as st
import pandas as pd
import joblib

# ===========================
# Manual Encoding Dictionaries & Functions
# ===========================

def encode_relevent_experience(value):
    if value == 'Has relevent experience':
        return 1
    else:
        return 0
    
def encode_enrolled_university(value):
    if value == 'no_enrollment':
        return 1
    elif value == 'Full time course':
        return 2
    elif value == 'Part time course':
        return 3
    else:
        return 0
    
def encode_major_discipline(value):
    if value == 'STEM':
        return 1
    elif value == 'Business Degree':
        return 2
    elif value == 'Arts':
        return 3
    elif value == 'Humanities':
        return 4
    elif value == 'No Major':
        return 5
    elif value == 'Other':
        return 6
    else:
        return 0
    
def encode_company_size(value):
    if value == '<10':
        return 1
    elif value == '10/49':
        return 2
    elif value == '50-99':
        return 3
    elif value == '100-500':
        return 4
    elif value == '500-999':
        return 5
    elif value == '1000-4999':
        return 6
    elif value == '5000-9999':
        return 7
    elif value == '10000+':
        return 8
    else:
        return 0
    
def encode_company_type(value):
    if value == 'Pvt Ltd':
        return 1
    elif value == 'Funded Startup':
        return 2
    elif value == 'Public Sector':
        return 3
    elif value == 'NGO':
        return 4
    elif value == 'Early Stage Startup':
        return 5
    elif value == 'Other':
        return 6
    else:
        return 0
    
def encode_education_level(val):
    if val == 'Primary School':
        return 1
    elif val == 'High School':
        return 2
    elif val == 'Graduate':
        return 3
    elif val == 'Masters':
        return 4
    elif val == 'Phd':
        return 5
    else:
        return 0
    
# ===========================
# Load Trained Model & Scaler
# ===========================

model = joblib.load("grid_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===========================
# Streamlit UI
# ===========================

st.title("📊 Employee Retention Predictor")
st.markdown("Predict whether an employee is likely to stay or leave the company based on their profile.")

st.header("👤 Employee Information")

# Collecting Inputs
city_dev_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)

relevent_exp = st.selectbox("Relevant Experience", [
    'Has relevent experience',
    'No relevent experience'
])

education_level = st.selectbox("Education Level", [
    'Graduate', 'Masters', 'High School', 'Phd', 'Primary School', 'Unknown'
])

enrolled_university = st.selectbox("Enrolled University", [
    'no_enrollment', 'Full time course', 'Part time course', 'Unknown'
])

major_discipline = st.selectbox("Major Discipline", [
    'STEM', 'Business Degree', 'Arts', 'Other', 'Humanities', 'No Major', 'Unknown'
])

experience = st.selectbox("Years of Experience", [str(i) for i in range(0, 22)])

company_size = st.selectbox("Company Size", [
    '<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+', 'Unknown'
])

company_type = st.selectbox("Company Type", [
    'Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Public Sector', 'NGO', 'Other', 'Unknown'
])

last_new_job = st.selectbox("Years Since Last Job Change", ['0', '1', '2', '3', '4', '>4'])

# ===========================
# Encode & Prepare Data
# ===========================

input_dict = {
    'city_development_index': city_dev_index,
    'relevent_experience': encode_relevent_experience(relevent_exp),
    'education_level': encode_education_level(education_level),
    'enrolled_university': encode_enrolled_university(enrolled_university),
    'major_discipline': encode_major_discipline(major_discipline),
    'experience': int(experience),
    'company_size': encode_company_size(company_size),
    'company_type': encode_company_type(company_type),
    'last_new_job': 5 if last_new_job == '>4' else int(last_new_job)
}

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# ===========================
# Recreate the scaled input with original feature names to match model training structure
# ===========================

feature_names = [
    'city_development_index',
    'relevent_experience',
    'education_level',
    'enrolled_university',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job'
]

input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

# ===========================
# Prediction Output
# ===========================

if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled_df)[0]
    if prediction == 1:
        st.error("Prediction: The employee is **looking for job change**.")
    else:
        st.success("Prediction: The employee is **not looking for a job change**.")

# ===========================
# Reference: Encoding Documentation
# ===========================

with st.expander("ℹ️ View Feature Encoding Reference"):
    st.markdown("""
    **Relevant Experience**  
    - Has relevent experience = 1  
    - No relevent experience = 0  

    **Target Variable**  
    - 1 = Likely to Leave  
    - 0 = Likely to Stay  

    **Enrolled University**  
    - no_enrollment = 1  
    - Full time course = 2  
    - Part time course = 3  
    - Unknown = 0  

    **Major Discipline**  
    - STEM = 1  
    - Business Degree = 2  
    - Arts = 3  
    - Humanities = 4  
    - No Major = 5  
    - Other = 6  
    - Unknown = 0  

    **Company Size**  
    - <10 = 1  
    - 10-49 = 2  
    - 50-99 = 3  
    - 100-500 = 4  
    - 500-999 = 5  
    - 1000-4999 = 6  
    - 5000-9999 = 7  
    - 10000+ = 8  
    - Unknown = 0  

    **Company Type**  
    - Pvt Ltd = 1  
    - Funded Startup = 2  
    - Public Sector = 3  
    - NGO = 4  
    - Early Stage Startup = 5  
    - Other = 6  
    - Unknown = 0  

    **Education Level**  
    - Primary School = 1  
    - High School = 2  
    - Graduate = 3  
    - Masters = 4  
    - PhD = 5  
    - Unknown = 0  
    """)
