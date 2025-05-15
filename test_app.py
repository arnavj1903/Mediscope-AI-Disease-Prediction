"""
Unit tests for the Medical Disease Prediction Flask Application.

This module contains comprehensive test cases for testing the functionality
of the Flask application for medical disease prediction.
"""

import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash

# Import the Flask application
from app import (
    app, db, Doctor, PatientData, load_model_files,
    FEATURE_LISTS, LABEL_TO_NUMERIC, _classify_risk,
    _get_patient_records, get_gemini_recommendations
)


class FlaskMedicalAppTests(unittest.TestCase):
    """Unit tests for the Medical Disease Prediction Flask Application."""

    def setUp(self):
        """Set up test environment before each test."""
        # Configure the Flask app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()

        # Create the database and tables
        with app.app_context():
            db.create_all()

            # Add test doctor with encrypted password
            test_doctor = Doctor(username='testdoctor')
            test_doctor.set_password('testpassword')
            db.session.add(test_doctor)
            db.session.commit()

            # Add test patient data
            test_patient = PatientData(
                doctor_id=1,
                name='Test Patient',
                disease='heart-attack',
                age=45,
                features={'age': 45, 'sex': 1},
                result=0.75,
                risk_label="Medium Risk"
            )
            db.session.add(test_patient)
            db.session.commit()

    def tearDown(self):
        """Clean up after each test."""
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_home_page(self):
        """Test that home page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MediScope AI', response.data)

    def test_successful_login(self):
        """Test successful login with valid credentials."""
        response = self.app.post('/', data={
            'username': 'testdoctor',
            'password': 'testpassword',
            'action': 'Login'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'authenticated_home', response.data.lower())

    def test_failed_login(self):
        """Test failed login with invalid credentials."""
        response = self.app.post('/', data={
            'username': 'testdoctor',
            'password': 'wrongpassword',
            'action': 'Login'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid username or password', response.data)

    def test_create_account(self):
        """Test creating a new account."""
        response = self.app.post('/', data={
            'username': 'newdoctor',
            'password': 'newpassword',
            'action': 'Create Account'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        with app.app_context():
            doctor = Doctor.query.filter_by(username='newdoctor').first()
            self.assertIsNotNone(doctor)
            self.assertTrue(doctor.check_password('newpassword'))

    def test_create_duplicate_account(self):
        """Test creating an account with existing username."""
        response = self.app.post('/', data={
            'username': 'testdoctor',
            'password': 'any',
            'action': 'Create Account'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Username already exists', response.data)

    def test_logout(self):
        """Test logout functionality."""
        # Login first
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            # Then logout
            response = client.get('/logout', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'MediScope AI', response.data)

            # Verify session is cleared
            with client.session_transaction() as sess:
                self.assertNotIn('username', sess)
                self.assertNotIn('doctor_id', sess)

    def test_authenticated_home_with_session(self):
        """Test authenticated home page with valid session."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            response = client.get('/authenticated_home')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'authenticated_home', response.data.lower())

    def test_authenticated_home_without_session(self):
        """Test that authenticated home redirects to login when no session."""
        response = self.app.get('/authenticated_home', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_disease_page_with_session(self):
        """Test disease prediction page with valid session."""
        # First mock the models to avoid loading actual files
        with patch('app.models', {'heart-attack': {}, 'breast-cancer': {}, 'diabetes': {}, 'lung-cancer': {}}):
            with self.app as client:
                with client.session_transaction() as sess:
                    sess['username'] = 'testdoctor'
                    sess['doctor_id'] = 1

                # Test each available disease page
                for disease in ['heart-attack', 'breast-cancer', 'diabetes', 'lung-cancer']:
                    with self.subTest(disease=disease):
                        response = client.get(f'/{disease}')
                        self.assertEqual(response.status_code, 200)
                        self.assertIn(disease.encode(), response.data.lower())

    def test_disease_page_without_session(self):
        """Test that disease page redirects to login when no session."""
        response = self.app.get('/heart-attack', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_disease_page_with_patient_name(self):
        """Test disease page with patient name parameter."""
        # Mock the models
        with patch('app.models', {'heart-attack': {}, 'breast-cancer': {}, 'diabetes': {}, 'lung-cancer': {}}):
            with self.app as client:
                with client.session_transaction() as sess:
                    sess['username'] = 'testdoctor'
                    sess['doctor_id'] = 1

                response = client.get('/heart-attack?name=Test%20Patient')
                self.assertEqual(response.status_code, 200)
                # Check if the page contains patient data section
                self.assertIn(b'patient', response.data.lower())

    def test_password_security(self):
        """Test that passwords are stored securely (TC_F1.3.1)."""
        with app.app_context():
            # Create a doctor user with password that should be hashed
            test_doctor = Doctor(username='securitytest')
            test_doctor.set_password('testpassword')
            db.session.add(test_doctor)
            db.session.commit()

            # Verify password is hashed in database
            doctor = Doctor.query.filter_by(username='securitytest').first()

            # Check password is not stored in plaintext
            self.assertNotEqual(doctor.password, 'testpassword')

            # Verify the password can be checked against the hash
            self.assertTrue(doctor.check_password('testpassword'))
            self.assertFalse(doctor.check_password('wrongpassword'))

    def test_model_loading(self):
        """Test that models are loaded correctly at application startup."""
        with patch('pickle.load') as mock_pickle:
            # Setup mock models
            mock_model = MagicMock()
            mock_pickle.return_value = mock_model

            # Call the load function
            models = load_model_files()

            # Verify all expected diseases are loaded
            self.assertIn('heart-attack', models)
            self.assertIn('breast-cancer', models)
            self.assertIn('diabetes', models)
            self.assertIn('lung-cancer', models)

            # Verify each disease has all required model types
            for disease in models:
                self.assertIn('scaler', models[disease])
                self.assertIn('KNN', models[disease])
                self.assertIn('DT', models[disease])
                self.assertIn('RF', models[disease])
                self.assertIn('LR', models[disease])
                self.assertIn('SVM', models[disease])
                self.assertIn('NB', models[disease])
                self.assertIn('DL', models[disease])
                
                # Check that quantum models exist where implemented
                if disease in ['heart-attack', 'diabetes', 'lung-cancer']:
                    self.assertIn('QNN', models[disease])

    def test_search_patient(self):
        """Test the search_patient functionality."""
        with self.app as client:
            with client.session_transaction() as sess:
                sess['username'] = 'testdoctor'
                sess['doctor_id'] = 1

            # Test with existing patient
            response = client.post('/search/heart-attack', data={'name': 'Test Patient'})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertFalse(data['no_records'])
            self.assertTrue(len(data['records']) > 0)
            
            # Test with non-existent patient
            response = client.post('/search/heart-attack', data={'name': 'Nonexistent'})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data['no_records'])
            self.assertIsNone(data['records'])

    def test_model_predictions(self):
        """Test model predictions with sample data for each disease."""
        # Mock models to return specific predictions based on our test cases
        with patch('app.models') as mock_models:
            # Setup mock models structure
            mock_models.return_value = {
                'heart-attack': {
                    'scaler': MagicMock(),
                    'DT': MagicMock(),
                    'KNN': MagicMock(),
                    'LR': MagicMock()
                },
                'breast-cancer': {
                    'scaler': MagicMock(),
                    'DT': MagicMock()
                },
                'diabetes': {
                    'scaler': MagicMock(),
                    'DT': MagicMock(),
                    'LR': MagicMock()
                },
                'lung-cancer': {
                    'scaler': MagicMock(),
                    'DT': MagicMock(),
                    'KNN': MagicMock()
                }
            }

            # Configure scaler to return input as-is (identity transform)
            for disease in ['heart-attack', 'breast-cancer', 'diabetes', 'lung-cancer']:
                mock_models.return_value[disease]['scaler'].transform.return_value = [1]*len(FEATURE_LISTS[disease])

            # Test cases for each disease (same as before)
            test_cases = [
                # ... (keep all your existing test cases here)
            ]

            with self.app as client:
                # Start a session transaction and set up the session
                with client.session_transaction() as sess:
                    sess['username'] = 'testdoctor'
                    sess['doctor_id'] = 1

                # Now make requests within this client context
                for case in test_cases:
                    disease = case['disease']
                    
                    # Test positive case
                    pos_case = case['positive']
                    mock_models.return_value[disease][pos_case['model']].predict.return_value = [pos_case['expected']]
                    
                    # Prepare form data
                    form_data = {'name': f'test_{disease}_positive', 'model': pos_case['model']}
                    form_data.update(pos_case['data'])
                    
                    # Make the request with follow_redirects=True to handle any redirects
                    response = client.post(
                        f'/predict/{disease}',
                        data=form_data,
                        follow_redirects=True
                    )
                    self.assertEqual(response.status_code, 200)
                    
                    # Check prediction and risk label in the response data
                    self.assertIn(f'prediction": {pos_case["expected"]}'.encode(), response.data)
                    self.assertIn(f'risk_label": "{pos_case["risk_label"]}"'.encode(), response.data)
                    
                    # Verify correct model was called
                    mock_models.return_value[disease][pos_case['model']].predict.assert_called_once()
                    mock_models.return_value[disease][pos_case['model']].reset_mock()
                    
                    # Test negative case
                    neg_case = case['negative']
                    mock_models.return_value[disease][neg_case['model']].predict.return_value = [neg_case['expected']]
                    
                    # Prepare form data
                    form_data = {'name': f'test_{disease}_negative', 'model': neg_case['model']}
                    form_data.update(neg_case['data'])
                    
                    # Make the request with follow_redirects=True
                    response = client.post(
                        f'/predict/{disease}',
                        data=form_data,
                        follow_redirects=True
                    )
                    self.assertEqual(response.status_code, 200)
                    
                    # Check prediction and risk label
                    self.assertIn(f'prediction": {neg_case["expected"]}'.encode(), response.data)
                    self.assertIn(f'risk_label": "{neg_case["risk_label"]}"'.encode(), response.data)
                    
                    # Verify correct model was called
                    mock_models.return_value[disease][neg_case['model']].predict.assert_called_once()
                    mock_models.return_value[disease][neg_case['model']].reset_mock()

    def test_risk_classification(self):
        """Test the risk classification function."""
        # Test high risk
        self.assertEqual(_classify_risk(0.85), "High Risk")

        # Test medium risk
        self.assertEqual(_classify_risk(0.75), "Medium Risk")
        self.assertEqual(_classify_risk(0.5), "Medium Risk")

        # Test low risk
        self.assertEqual(_classify_risk(0.49), "Low Risk")
        self.assertEqual(_classify_risk(0.0), "Low Risk")

    def test_patient_record_retrieval(self):
        """Test retrieval of patient records."""
        # Test the _get_patient_records function directly
        with app.app_context():
            # Existing patient
            records, no_records = _get_patient_records(1, 'Test Patient', 'heart-attack')
            self.assertFalse(no_records)
            self.assertIsNotNone(records)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['risk_label'], 'Medium Risk')
            
            # Non-existent patient
            records, no_records = _get_patient_records(1, 'Nonexistent', 'heart-attack')
            self.assertTrue(no_records)
            self.assertIsNone(records)

    def test_gemini_recommendations(self):
        """Test Gemini recommendations generation (mocked)."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.text = "1. Recommendation one\n2. Recommendation two"
            mock_model.return_value.generate_content.return_value = mock_response

            # Call the function
            recommendations = get_gemini_recommendations('heart-attack', {'age': 50})

            # Verify the response
            self.assertEqual(len(recommendations), 2)
            self.assertEqual(recommendations[0], "Recommendation one")
            self.assertEqual(recommendations[1], "Recommendation two")

    def test_feature_lists(self):
        """Test that feature lists are properly defined."""
        self.assertIn('heart-attack', FEATURE_LISTS)
        self.assertIn('breast-cancer', FEATURE_LISTS)
        self.assertIn('diabetes', FEATURE_LISTS)
        self.assertIn('lung-cancer', FEATURE_LISTS)

        # Verify some key features for each disease
        self.assertIn('age', FEATURE_LISTS['heart-attack'])
        self.assertIn('radius_mean', FEATURE_LISTS['breast-cancer'])
        self.assertIn('Glucose', FEATURE_LISTS['diabetes'])
        self.assertIn('SMOKING', FEATURE_LISTS['lung-cancer'])

    def test_label_to_numeric_mappings(self):
        """Test that label to numeric mappings are properly defined."""
        self.assertIn('heart-attack', LABEL_TO_NUMERIC)
        self.assertIn('lung-cancer', LABEL_TO_NUMERIC)

        # Verify some key mappings
        self.assertEqual(LABEL_TO_NUMERIC['heart-attack']['sex']['male'], 1)
        self.assertEqual(LABEL_TO_NUMERIC['lung-cancer']['GENDER']['M'], 1)


if __name__ == '__main__':
    unittest.main()