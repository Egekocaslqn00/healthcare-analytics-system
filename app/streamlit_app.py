"""
Healthcare Analytics System - Streamlit Dashboard
Multi-Disease Prediction Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        models['heart'] = joblib.load('../models/heart_disease_model.pkl')
        models['heart_scaler'] = joblib.load('../models/heart_scaler.pkl')
        models['diabetes'] = joblib.load('../models/diabetes_model.pkl')
        models['diabetes_scaler'] = joblib.load('../models/diabetes_scaler.pkl')
        models['stroke'] = joblib.load('../models/stroke_model.pkl')
        models['stroke_scaler'] = joblib.load('../models/stroke_scaler.pkl')
        models['stroke_encoders'] = joblib.load('../models/stroke_encoders.pkl')
    except:
        st.error("⚠️ Models not found. Please train the models first.")
        return None
    return models

models = load_models()

# Header
st.markdown('<p class="main-header">🏥 Healthcare Analytics System</p>', unsafe_allow_html=True)
st.markdown("### Multi-Disease Prediction Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/health-checkup.png", width=100)
    st.title("Navigation")
    page = st.radio("Select Disease", 
                    ["🏠 Home", "❤️ Heart Disease", "🩺 Diabetes", "🧠 Stroke", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This application uses machine learning models trained on real healthcare datasets to predict disease risk.
    
    **Datasets:**
    - Heart Disease (UCI)
    - Pima Indians Diabetes
    - Stroke Prediction
    
    **Models:**
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    - XGBoost
    """)

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ❤️ Heart Disease")
        st.metric("Model Accuracy", "82%")
        st.metric("ROC-AUC Score", "0.91")
        st.info("Predicts heart disease risk based on 13 clinical features")
    
    with col2:
        st.markdown("### 🩺 Diabetes")
        st.metric("Model Accuracy", "76%")
        st.metric("ROC-AUC Score", "0.83")
        st.info("Predicts diabetes risk based on 8 diagnostic measurements")
    
    with col3:
        st.markdown("### 🧠 Stroke")
        st.metric("Model Accuracy", "75%")
        st.metric("ROC-AUC Score", "0.84")
        st.info("Predicts stroke risk based on 10 health parameters")
    
    st.markdown("---")
    st.markdown("### 📈 Project Highlights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **✅ Key Features:**
        - Real healthcare datasets
        - Multiple ML algorithms
        - Comprehensive EDA
        - Model comparison & optimization
        - Interactive predictions
        - Feature importance analysis
        """)
    
    with col2:
        st.warning("""
        **⚠️ Disclaimer:**
        This tool is for educational and research purposes only.
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always consult with qualified healthcare providers.
        """)

# Heart Disease Prediction
elif page == "❤️ Heart Disease":
    st.markdown('<p class="sub-header">❤️ Heart Disease Risk Prediction</p>', unsafe_allow_html=True)
    
    if models is None:
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    
    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    if st.button("🔍 Predict Heart Disease Risk", key="heart_predict"):
        # Prepare input
        sex_val = 1 if sex == "Male" else 0
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
        exang_val = 1 if exang == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
        
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                               thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
        
        # Scale and predict
        input_scaled = models['heart_scaler'].transform(input_data)
        prediction = models['heart'].predict(input_scaled)[0]
        probability = models['heart'].predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        risk_prob = probability[1] * 100
        
        if risk_prob > 70:
            risk_class = "high-risk"
            risk_level = "HIGH RISK ⚠️"
        elif risk_prob > 40:
            risk_class = "medium-risk"
            risk_level = "MEDIUM RISK ⚡"
        else:
            risk_class = "low-risk"
            risk_level = "LOW RISK ✅"
        
        st.markdown(f'<div class="prediction-box {risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_level}")
        st.markdown(f"**Heart Disease Probability: {risk_prob:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if risk_prob > 70 else "orange" if risk_prob > 40 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Diabetes Prediction
elif page == "🩺 Diabetes":
    st.markdown('<p class="sub-header">🩺 Diabetes Risk Prediction</p>', unsafe_allow_html=True)
    
    if models is None:
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dl)", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=20, max_value=100, value=30)
    
    if st.button("🔍 Predict Diabetes Risk", key="diabetes_predict"):
        # Prepare input
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, dpf, age]])
        
        # Scale and predict
        input_scaled = models['diabetes_scaler'].transform(input_data)
        prediction = models['diabetes'].predict(input_scaled)[0]
        probability = models['diabetes'].predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        risk_prob = probability[1] * 100
        
        if risk_prob > 70:
            risk_class = "high-risk"
            risk_level = "HIGH RISK ⚠️"
        elif risk_prob > 40:
            risk_class = "medium-risk"
            risk_level = "MEDIUM RISK ⚡"
        else:
            risk_class = "low-risk"
            risk_level = "LOW RISK ✅"
        
        st.markdown(f'<div class="prediction-box {risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_level}")
        st.markdown(f"**Diabetes Probability: {risk_prob:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if risk_prob > 70 else "orange" if risk_prob > 40 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Stroke Prediction
elif page == "🧠 Stroke":
    st.markdown('<p class="sub-header">🧠 Stroke Risk Prediction</p>', unsafe_allow_html=True)
    
    if models is None:
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0, max_value=100, value=50)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose = st.number_input("Average Glucose Level (mg/dl)", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    if st.button("🔍 Predict Stroke Risk", key="stroke_predict"):
        # Encode inputs
        gender_encoded = models['stroke_encoders']['gender'].transform([gender])[0]
        married_encoded = models['stroke_encoders']['ever_married'].transform([ever_married])[0]
        work_encoded = models['stroke_encoders']['work_type'].transform([work_type])[0]
        residence_encoded = models['stroke_encoders']['Residence_type'].transform([residence_type])[0]
        smoking_encoded = models['stroke_encoders']['smoking_status'].transform([smoking_status])[0]
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0
        
        # Prepare input
        input_data = np.array([[gender_encoded, age, hypertension_val, heart_disease_val, 
                               married_encoded, work_encoded, residence_encoded, 
                               avg_glucose, bmi, smoking_encoded]])
        
        # Scale and predict
        input_scaled = models['stroke_scaler'].transform(input_data)
        prediction = models['stroke'].predict(input_scaled)[0]
        probability = models['stroke'].predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        risk_prob = probability[1] * 100
        
        if risk_prob > 70:
            risk_class = "high-risk"
            risk_level = "HIGH RISK ⚠️"
        elif risk_prob > 40:
            risk_class = "medium-risk"
            risk_level = "MEDIUM RISK ⚡"
        else:
            risk_class = "low-risk"
            risk_level = "LOW RISK ✅"
        
        st.markdown(f'<div class="prediction-box {risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_level}")
        st.markdown(f"**Stroke Probability: {risk_prob:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if risk_prob > 70 else "orange" if risk_prob > 40 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Analytics Page
elif page == "📊 Analytics":
    st.markdown('<p class="sub-header">📊 Model Performance Analytics</p>', unsafe_allow_html=True)
    
    # Load results
    try:
        heart_results = pd.read_csv('../reports/heart_disease_model_results.csv')
        diabetes_results = pd.read_csv('../reports/diabetes_model_results.csv')
        stroke_results = pd.read_csv('../reports/stroke_model_results.csv')
        
        tab1, tab2, tab3 = st.tabs(["Heart Disease", "Diabetes", "Stroke"])
        
        with tab1:
            st.markdown("### Heart Disease Model Comparison")
            st.dataframe(heart_results, use_container_width=True)
            
            fig = px.bar(heart_results, x='Model', y='ROC-AUC', 
                        title='Model ROC-AUC Comparison',
                        color='ROC-AUC', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Diabetes Model Comparison")
            st.dataframe(diabetes_results, use_container_width=True)
            
            fig = px.bar(diabetes_results, x='Model', y='ROC-AUC', 
                        title='Model ROC-AUC Comparison',
                        color='ROC-AUC', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Stroke Model Comparison")
            st.dataframe(stroke_results, use_container_width=True)
            
            fig = px.bar(stroke_results, x='Model', y='ROC-AUC', 
                        title='Model ROC-AUC Comparison',
                        color='ROC-AUC', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Healthcare Analytics System | Built with Streamlit & Scikit-learn</p>
    <p>⚠️ For Educational Purposes Only - Not for Medical Use</p>
</div>
""", unsafe_allow_html=True)
