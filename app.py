"""
MediScope AI - Medical Disease Prediction Application.

This module contains a Flask application for medical disease prediction
using machine learning models.
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import os
from dotenv import load_dotenv
import logging
import numpy as np
from uuid import uuid4
import google.generativeai as genai
import requests
import re
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid4())  # Use a random UUID for security
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAngvpflodR7WNa7BjEneKwhlQvx8Nvx9Q")
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found in .env file")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")


def load_model_files():
    """Load all machine learning models and scalers from files."""
    models_dict = {
        'heart-attack': {
            'scaler': pickle.load(open('heart_attack/heart_scaler', 'rb')),
            'KNN': pickle.load(open('heart_attack/KNN_model', 'rb')),
            'DT': pickle.load(open('heart_attack/DT_model', 'rb')),
            'RF': pickle.load(open('heart_attack/RF_model', 'rb')),
            'LR': pickle.load(open('heart_attack/LR_model', 'rb')),
            'SVM': pickle.load(open('heart_attack/SVM_model', 'rb')),
            'NB': pickle.load(open('heart_attack/NB_model', 'rb')),
            'DL': pickle.load(open('heart_attack/DL_model', 'rb')),
            'QNN' : pickle.load(open('heart_attack/qnn_model.pkl', 'rb'))
        },
        'breast-cancer': {
            'scaler': pickle.load(open('breast_cancer/breast_scaler', 'rb')),
            'KNN': pickle.load(open('breast_cancer/KNN_model', 'rb')),
            'DT': pickle.load(open('breast_cancer/DT_model', 'rb')),
            'RF': pickle.load(open('breast_cancer/RF_model', 'rb')),
            'LR': pickle.load(open('breast_cancer/LR_model', 'rb')),
            'SVM': pickle.load(open('breast_cancer/SVM_model', 'rb')),
            'NB': pickle.load(open('breast_cancer/NB_model', 'rb')),
            'DL': pickle.load(open('breast_cancer/DL_model', 'rb'))
        },
        'diabetes': {
            'scaler': pickle.load(open('diabetes/diabetes_scaler', 'rb')),
            'KNN': pickle.load(open('diabetes/KNN_model', 'rb')),
            'DT': pickle.load(open('diabetes/DT_model', 'rb')),
            'RF': pickle.load(open('diabetes/RF_model', 'rb')),
            'LR': pickle.load(open('diabetes/LR_model', 'rb')),
            'SVM': pickle.load(open('diabetes/SVM_model', 'rb')),
            'NB': pickle.load(open('diabetes/NB_model', 'rb')),
            'DL': pickle.load(open('diabetes/DL_model', 'rb')),
            'QNN' : pickle.load(open('diabetes/qnn_model.pkl', 'rb'))
        },
        'lung-cancer': {
            'scaler': pickle.load(open('lung_cancer/lung_scaler', 'rb')),
            'KNN': pickle.load(open('lung_cancer/KNN_model', 'rb')),
            'DT': pickle.load(open('lung_cancer/DT_model', 'rb')),
            'RF': pickle.load(open('lung_cancer/RF_model', 'rb')),
            'LR': pickle.load(open('lung_cancer/LR_model', 'rb')),
            'SVM': pickle.load(open('lung_cancer/SVM_model', 'rb')),
            'NB': pickle.load(open('lung_cancer/NB_model', 'rb')),
            'DL': pickle.load(open('lung_cancer/DL_model', 'rb')),
            'QNN' : pickle.load(open('lung_cancer/qnn_model.pkl', 'rb'))
        }
    }
    logger.info("Models and scalers loaded successfully")
    return models_dict


# Load the models at application startup
try:
    models = load_model_files()
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    models = {}


# Database Models
class Doctor(db.Model):
    """Doctor model for storing physician login data."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)  # Increased length for hash

    def set_password(self, password):
        """Set password hash."""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Check password against stored hash."""
        return check_password_hash(self.password, password)


class PatientData(db.Model):
    """PatientData model for storing patient health information."""
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    disease = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    features = db.Column(db.PickleType, nullable=False)
    result = db.Column(db.Float, nullable=True)
    risk_label = db.Column(db.String(20), nullable=True)


# Initialize database
with app.app_context():
    db.create_all()
    logger.info("Database initialized successfully")


# Define feature lists for each disease
FEATURE_LISTS = {
    'heart-attack': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'breast-cancer': [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ],
    'diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    'lung-cancer': [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
        'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
}


# Mapping of human-readable labels to numeric values
LABEL_TO_NUMERIC = {
    'heart-attack': {
        'sex': {'female': 0, 'male': 1},
        'cp': {'typical_angina': 0, 'atypical_angina': 1,
               'non_anginal_pain': 2, 'asymptomatic': 3},
        'fbs': {'false': 0, 'true': 1},
        'restecg': {'normal': 0, 'st_t_abnormality': 1, 'lv_hypertrophy': 2},
        'exang': {'no': 0, 'yes': 1},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
        'ca': {'0': 0, '1': 1, '2': 2, '3': 3},
        'thal': {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2, 'other': 3}
    },
    'breast-cancer': {
        'diagnosis': {'B': 0, 'M': 1}
    },
    'diabetes': {},
    'lung-cancer': {
        'GENDER': {'F': 0, 'M': 1},
        'SMOKING': {'1': 0, '2': 1},
        'YELLOW_FINGERS': {'1': 0, '2': 1},
        'ANXIETY': {'1': 0, '2': 1},
        'PEER_PRESSURE': {'1': 0, '2': 1},
        'CHRONIC_DISEASE': {'1': 0, '2': 1},
        'FATIGUE': {'1': 0, '2': 1},
        'ALLERGY': {'1': 0, '2': 1},
        'WHEEZING': {'1': 0, '2': 1},
        'ALCOHOL_CONSUMING': {'1': 0, '2': 1},
        'COUGHING': {'1': 0, '2': 1},
        'SHORTNESS_OF_BREATH': {'1': 0, '2': 1},
        'SWALLOWING_DIFFICULTY': {'1': 0, '2': 1},
        'CHEST_PAIN': {'1': 0, '2': 1}
    }
}


def get_gemini_recommendations(disease, patient_data):
    """Get recommendations from Gemini API based on disease and patient data."""
    try:
        # Prepare the prompt
        prompt = f"""
        The patient has tested positive for {disease.replace('-', ' ')}.
        Patient details: Age: {patient_data.get('age', 'unknown')}.
        Provide exactly 5 concise, actionable recommendations for managing or treating this condition.
        Each recommendation should be a single sentence.
        Write in plain text, without markdown (no *, **, -, or numbered lists like 1.).
        Do not include disclaimers or references to 'features' or 'healthcare providers.'
        """

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        # Parse response
        recommendations = response.text.split('\n')[:5]  # Limit to 5 recommendations
        recommendations = [r.strip() for r in recommendations if r.strip()]  # Clean up
        cleaned_recommendations = []
        for rec in recommendations:
            # Remove markdown list markers
            cleaned = re.sub(r'^\s*[\*\-]\s*|^[0-9]+\.\s*', '', rec).strip()
            if cleaned:  # Only add non-empty recommendations
                cleaned_recommendations.append(cleaned)

        recommendations = cleaned_recommendations[:5]
        if not recommendations:
            raise ValueError("No valid recommendations received from Gemini API")

        logger.info(f"Gemini API provided recommendations for {disease}")
        return recommendations
    except Exception as e:
        logger.error(f"Gemini API error for {disease}: {str(e)}")
        return ["Unable to retrieve recommendations at this time."]


def _get_patient_records(doctor_id, name, disease):
    """Get patient records for a given doctor, name, and disease."""
    records = PatientData.query.filter_by(
        doctor_id=doctor_id, name=name, disease=disease
    ).all()

    if records:
        return [{
            'id': r.id,
            'age': r.age,
            'features': r.features,
            'result': r.result,
            'risk_label': r.risk_label
        } for r in records], False
    return None, True


def _classify_risk(prediction):
    """Classify risk based on prediction value."""
    if prediction >= 0.8:
        return "High Risk"
    elif prediction >= 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"


@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon requests."""
    return '', 204


@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page route handler with login functionality."""
    if 'username' in session:
        return redirect(url_for('authenticated_home'))

    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form['username']
        password = request.form['password']

        if action == 'Login':
            doctor = Doctor.query.filter_by(username=username).first()
            if doctor and doctor.check_password(password):
                session['username'] = doctor.username
                session['doctor_id'] = doctor.id
                logger.info(f"Doctor {username} logged in successfully")
                return redirect(url_for('authenticated_home'))
            else:
                error = "Invalid username or password"
                logger.warning(f"Failed login attempt for username: {username}")
        elif action == 'Create Account':
            if Doctor.query.filter_by(username=username).first():
                error = "Username already exists"
                logger.warning(f"Attempt to create existing username: {username}")
            else:
                new_doctor = Doctor(username=username)
                new_doctor.set_password(password)
                db.session.add(new_doctor)
                db.session.commit()
                session['username'] = username
                session['doctor_id'] = new_doctor.id
                logger.info(f"New doctor account created: {username}")
                return redirect(url_for('authenticated_home'))

    return render_template('home.html', error=error)

@app.route('/authenticated_home')
def authenticated_home():
    """Authenticated home page route handler."""
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('authenticated_home.html')

@app.route('/logout')
def logout():
    """Logout route handler."""
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('home'))


@app.route('/<disease>')
def disease_page(disease):
    """Disease prediction page route handler."""
    if 'username' not in session:
        return redirect(url_for('home'))

    if disease not in models:
        return redirect(url_for('authenticated_home'))

    name = request.args.get('name', '')
    records = None
    no_records = False

    if name:
        doctor_id = session.get('doctor_id')
        records, no_records = _get_patient_records(doctor_id, name, disease)

    return render_template(
        f'{disease}.html',
        prediction=None,
        name=name,
        error=None,
        records=records,
        no_records=no_records
    )


@app.route('/search/<disease>', methods=['POST'])
def search_patient(disease):
    """Search for patient records."""
    if 'username' not in session:
        return jsonify({'records': None, 'no_records': True})

    name = request.form.get('name')
    if not name:
        return jsonify({'records': None, 'no_records': True})

    doctor_id = session.get('doctor_id')
    if not doctor_id:
        return jsonify({'records': None, 'no_records': True})

    records, no_records = _get_patient_records(doctor_id, name, disease)
    
    print()
    print(f"patient records of {name}", records)
    print()

    return jsonify({
        'records': records,
        'no_records': no_records
    })


@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    """Make disease prediction based on input features."""
    if 'username' not in session:
        return redirect(url_for('login'))

    if disease not in models:
        return redirect(url_for('authenticated_home'))

    doctor_id = session.get('doctor_id')
    doctor = Doctor.query.filter_by(id=doctor_id).first()
    model_type = request.form.get('model', 'RF')  # Default to Random Forest
    name = request.form.get('name', '')

    # Extract and validate features
    features = []
    age = None
    error_message = None

    #checking input data
    print()
    print("input data", request.form)
    print()

    for feature in FEATURE_LISTS[disease]:
        value = request.form.get(feature)
        if not value:
            error_message = f"Missing value for {feature}"
            break

        if disease in LABEL_TO_NUMERIC and feature in LABEL_TO_NUMERIC[disease]:
            value = LABEL_TO_NUMERIC[disease][feature].get(value, None)
            if value is None:
                error_message = f"Invalid value for {feature}"
                break
        else:
            try:
                value = float(value)
            except ValueError:
                error_message = f"Invalid value for {feature}"
                break

        features.append(value)
        if feature.lower() in ['age', 'AGE']:
            age = int(float(value))  # Convert to float first, then to int

    # If there was an error in feature extraction
    if error_message:
        records, no_records = _get_patient_records(doctor_id, name, disease) if name else (None, False)
        return render_template(
            f'{disease}.html',
            prediction=None,
            name=name,
            error=error_message,
            records=records,
            no_records=no_records
        )

    # Scale features
    try:
        scaler = models[disease]['scaler']
        scaled_features = scaler.transform([features])[0]
        logger.info(f"Features scaled for {disease} prediction")
    except Exception as e:
        logger.error(f"Scaling error for {disease}: {str(e)}")
        records, no_records = _get_patient_records(doctor_id, name, disease) if name else (None, False)
        return render_template(
            f'{disease}.html',
            prediction=None,
            name=name,
            error="Error in scaling features",
            records=records,
            no_records=no_records
        )

    # Make prediction
    try:
        if model_type == 'DL':
            model_input = np.array([scaled_features])
            prediction = models[disease][model_type].predict(model_input)[0]
            prediction = float(prediction)  # Convert to float for consistent type
        else:
            model_input = [scaled_features]
            prediction = models[disease][model_type].predict(model_input)[0]

        prediction = float(prediction)  # Convert to float for consistent type
        prediction = 1.0 if prediction >= 0.5 else 0.0

        logger.info(f"Prediction for {name} with {model_type}: {prediction}")
    except Exception as e:
        logger.error(f"Prediction error for {disease} with {model_type}: {str(e)}")
        records, no_records = _get_patient_records(doctor_id, name, disease) if name else (None, False)
        return render_template(
            f'{disease}.html',
            prediction=None,
            name=name,
            error="Error in prediction",
            records=records,
            no_records=no_records
        )

    # Get recommendations for positive predictions
    recommendations = None
    risk_label = _classify_risk(prediction)

    if prediction >= 0.5:  # Consider as positive prediction
        patient_data = {'age': age, 'features': features}
        recommendations = get_gemini_recommendations(disease, patient_data)

    # Store or update patient data
    feature_dict = {feature: value for feature, value in zip(FEATURE_LISTS[disease], features)}

    print()
    print(f"Saving patient data for {name}, disease: {disease}, doctor_id: {doctor_id}, age: {age}")
    print()

    if name:
        try:
            # Check if patient exists
            patient = PatientData.query.filter_by(
                doctor_id=doctor_id, name=name, disease=disease, age=age
            ).first()

            if patient:
                print(f"Updating existing patient record: {patient.id}")
                patient.features = feature_dict
                patient.result = float(prediction)
                patient.risk_label = risk_label
            else:
                print(f"Creating new patient record for {name}")
                patient = PatientData(
                    doctor_id=doctor_id,
                    name=name,
                    disease=disease,
                    age=age,
                    features=feature_dict,
                    result=float(prediction),
                    risk_label=risk_label
                )
                db.session.add(patient)
            
            db.session.commit()
            print(f"Database commit successful for {name}")
            
            # Verify the record was saved
            verification = PatientData.query.filter_by(
                doctor_id=doctor_id, name=name, disease=disease, age=age
            ).first()
            
            if verification:
                print(f"Verified record exists with ID: {verification.id}")
            else:
                print("WARNING: Failed to verify record exists after save!")
                
        except Exception as e:
            print(f"Error saving patient data: {str(e)}")
            db.session.rollback()
            logger.error(f"Database error: {str(e)}")

    # Fetch records for display
    records, no_records = _get_patient_records(doctor_id, name, disease) if name else (None, False)

    return render_template(
        f'{disease}.html',
        prediction=prediction,
        name=name,
        error=None,
        records=records,
        no_records=no_records,
        recommendations=recommendations,
        risk_label=risk_label
    )


if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
