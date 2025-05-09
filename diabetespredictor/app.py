from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os
import pickle
import numpy as np

# Get the absolute path to the diabetespredictor directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Flask app with explicit template folder
template_dir = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create all tables
with app.app_context():
    db.create_all()

# Load the trained model and scaler
model_dir = os.path.join(os.path.dirname(BASE_DIR), 'AppModelTraining', 'models')
model_path = os.path.join(model_dir, 'diabetes_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    raise FileNotFoundError("Model or scaler file not found. Please ensure the model is trained first.")

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get and validate input values
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])

            # Print input values for debugging
            print(f"Input values: Glucose={glucose}, BP={blood_pressure}, Skin={skin_thickness}, Insulin={insulin}, BMI={bmi}")

            # Validate ranges
            if not (50 <= glucose <= 300):
                flash('Glucose must be between 50-300 mg/dL')
                return redirect(url_for('predict'))
            if not (40 <= blood_pressure <= 180):
                flash('Blood pressure must be between 40-180 mmHg')
                return redirect(url_for('predict'))
            if not (7 <= skin_thickness <= 99):
                flash('Skin thickness must be between 7-99 mm')
                return redirect(url_for('predict'))
            if not (14 <= insulin <= 846):
                flash('Insulin must be between 14-846 μU/mL')
                return redirect(url_for('predict'))
            if not (10 <= bmi <= 60):
                flash('BMI must be between 10-60 kg/m²')
                return redirect(url_for('predict'))

            # Create input dataframe with only core clinical measurements
            input_df = pd.DataFrame({
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'Age': [35]
            })

            # Ensure columns are in the same order as training data
            feature_columns = [
                'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'
            ]

            # Scale the features
            scaled_features = scaler.transform(input_df[feature_columns])
            
            # Make prediction using the model
            proba = model.predict_proba(scaled_features)[0][1] * 100
            prediction = model.predict(scaled_features)[0]

            # Print prediction results for debugging
            print(f"Raw prediction: {prediction}, Probability: {proba:.1f}%")

            # Define clinical thresholds based on medical guidelines
            thresholds = {
                'glucose': {
                    'normal': 100,  # Fasting glucose < 100 mg/dL is normal
                    'prediabetes': 126,  # Fasting glucose ≥ 126 mg/dL indicates diabetes
                    'high': 200  # Random glucose ≥ 200 mg/dL indicates diabetes
                },
                'bmi': {
                    'normal': 25,  # BMI < 25 is normal
                    'overweight': 30,  # BMI ≥ 30 is obese
                    'high': 35  # BMI ≥ 35 is severely obese
                },
                'insulin': {
                    'normal': 100,  # Normal fasting insulin
                    'high': 200,  # Elevated insulin
                    'very_high': 300  # Very high insulin
                },
                'blood_pressure': {
                    'normal': 120,  # Normal BP
                    'elevated': 140,  # Elevated BP
                    'high': 160  # High BP
                }
            }

            # Calculate individual risk scores (0-3 for each factor)
            risk_scores = {
                'glucose': (
                    3 if glucose >= thresholds['glucose']['high']
                    else 2 if glucose >= thresholds['glucose']['prediabetes']
                    else 1 if glucose >= thresholds['glucose']['normal']
                    else 0
                ),
                'bmi': (
                    3 if bmi >= thresholds['bmi']['high']
                    else 2 if bmi >= thresholds['bmi']['overweight']
                    else 1 if bmi >= thresholds['bmi']['normal']
                    else 0
                ),
                'insulin': (
                    3 if insulin >= thresholds['insulin']['very_high']
                    else 2 if insulin >= thresholds['insulin']['high']
                    else 1 if insulin >= thresholds['insulin']['normal']
                    else 0
                ),
                'blood_pressure': (
                    3 if blood_pressure >= thresholds['blood_pressure']['high']
                    else 2 if blood_pressure >= thresholds['blood_pressure']['elevated']
                    else 1 if blood_pressure >= thresholds['blood_pressure']['normal']
                    else 0
                )
            }

            # Calculate total risk score (0-12)
            total_risk = sum(risk_scores.values())
            
            # Calculate weighted risk score (0-100)
            weighted_risk = (
                (risk_scores['glucose'] * 25) +  # Glucose importance reduced
                (risk_scores['bmi'] * 20) +      # BMI importance reduced
                (risk_scores['insulin'] * 30) +  # Insulin importance increased
                (risk_scores['blood_pressure'] * 25)  # BP importance increased
            )

            # Combine model prediction with clinical assessment
            final_risk_score = (proba + weighted_risk) / 2

            # Determine risk level based on combined score
            if final_risk_score < 40:  # Increased threshold for Low Risk
                risk = "Low Risk"
                advice = "Low risk detected! Maintain your healthy lifestyle."
            elif final_risk_score < 70:  # Increased threshold for Medium Risk
                risk = "Medium Risk"
                advice = "Moderate risk. Consider lifestyle changes and regular check-ups."
            else:
                risk = "High Risk"
                advice = "High risk detected! Please consult a healthcare professional."

            # Prepare detailed analysis with specific ranges
            analysis = {
                'glucose_status': (
                    'High' if glucose >= thresholds['glucose']['high']
                    else 'Elevated' if glucose >= thresholds['glucose']['prediabetes']
                    else 'Normal'
                ),
                'bmi_status': (
                    'High' if bmi >= thresholds['bmi']['high']
                    else 'Elevated' if bmi >= thresholds['bmi']['overweight']
                    else 'Normal'
                ),
                'insulin_status': (
                    'High' if insulin >= thresholds['insulin']['very_high']
                    else 'Elevated' if insulin >= thresholds['insulin']['high']
                    else 'Normal'
                ),
                'bp_status': (
                    'High' if blood_pressure >= thresholds['blood_pressure']['high']
                    else 'Elevated' if blood_pressure >= thresholds['blood_pressure']['elevated']
                    else 'Normal'
                )
            }

            return render_template('predict.html',
                                prediction=risk,
                                probability=f"{final_risk_score:.1f}%",
                                advice=advice,
                                analysis=analysis)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")  # Debug logging
            flash(f'Error processing request: {str(e)}')
            return redirect(url_for('predict'))
    
    # For GET request, render template without prediction data
    return render_template('predict.html',
                         prediction=None,
                         probability=None,
                         advice=None,
                         analysis=None)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database initialized!")
    app.run(debug=True)