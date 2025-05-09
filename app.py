from flask import Flask, render_template, request, redirect, url_for, flash
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

# Create Flask app with explicit template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Changed to a fixed key for development
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'diabetes_model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError("Model file not found. Please train the model first.")

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if users already exist
        if User.query.count() == 0:
            # Create admin user
            admin = User(username='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            
            # Create test user
            test_user = User(username='test')
            test_user.set_password('test123')
            db.session.add(test_user)
            
            db.session.commit()
            print("Initial users created!")

# Routes
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Load diabetes data
    df = pd.read_csv('diabetes.csv')
    
    # Create statistics
    stats = {
        'total_patients': len(df),
        'diabetes_positive': len(df[df['Outcome'] == 1]),
        'diabetes_negative': len(df[df['Outcome'] == 0]),
        'avg_glucose': df['Glucose'].mean(),
        'avg_bmi': df['BMI'].mean()
    }
    
    # Create charts
    # Age distribution
    age_fig = px.histogram(df, x='Age', title='Age Distribution')
    age_chart = json.dumps(age_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # BMI vs Glucose scatter
    bmi_glucose_fig = px.scatter(df, x='BMI', y='Glucose', color='Outcome',
                                title='BMI vs Glucose by Diabetes Status')
    bmi_glucose_chart = json.dumps(bmi_glucose_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('dashboard.html', 
                         stats=stats,
                         age_chart=age_chart,
                         bmi_glucose_chart=bmi_glucose_chart)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            age = float(request.form['age'])

            # Create input array for prediction
            input_data = np.array([[glucose, blood_pressure, insulin, bmi, age]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]  # Get probability of positive class
            
            return render_template('predict.html', prediction=prediction, probability=probability)
        except Exception as e:
            flash(f'Error making prediction: {str(e)}')
            return render_template('predict.html', prediction=None, probability=None)
    
    return render_template('predict.html', prediction=None, probability=None)

# Initialize the database and create initial users
init_db()

if __name__ == '__main__':
    app.run(debug=True) 