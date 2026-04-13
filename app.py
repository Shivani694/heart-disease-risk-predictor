from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import numpy as np

import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_heart_disease_app' # Required for session

# Load model and scaler
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
else:
    model = None
    scaler = None
    print(f"Warning: {model_path} not found. Please train the model first.")


@app.route('/')
def welcome():
    # If already logged in, you can choose to skip welcome or let them see it.
    # Let's show the welcome page to everyone visiting root.
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_contact = request.form.get('contact')
        if user_contact:
            session['user'] = user_contact
            return redirect(url_for('dashboard'))
        return "Please enter a valid email or phone number.", 400
    
    # If already logged in, redirect to dashboard
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('welcome'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/form')
def form():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    if model is None or scaler is None:
        return "Model not found. Please run local setup scripts first.", 500
        
    if request.method == 'POST':
        try:
            # Extract data from the form
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            bp = int(request.form['bp'])
            cholesterol = int(request.form['cholesterol'])
            bs = int(request.form['bs'])
            hr = int(request.form['hr'])
            cp = int(request.form['cp'])
            angina = int(request.form['angina'])
            
            # Prepare data for prediction
            input_features = np.array([[age, gender, bp, cholesterol, bs, hr, cp, angina]])
            
            # Scale the features
            input_scaled = scaler.transform(input_features)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities) * 100
            
            # Map prediction back to labels
            risk_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
            color_map = {0: 'green', 1: 'yellow', 2: 'red'}
            recommendation_map = {
                0: 'Your heart health looks spectacular. Keep up the healthy lifestyle and exercise routines!',
                1: 'You display some subtle risk factors. Consider a general checkup and possibly improving your diet and exercise habits.',
                2: 'High risk detected based on your parameters. Please schedule a consultation with a cardiologist for a thorough examination.'
            }
            
            risk_level = risk_map[prediction]
            color_theme = color_map[prediction]
            recommendation = recommendation_map[prediction]

            # Generate Comparisions Status
            def get_status(val, norm_high, risk_high):
                if val <= norm_high: return "Normal", "green"
                elif val <= risk_high: return "Slightly High", "yellow"
                else: return "High Risk", "red"

            bp_status, bp_color = get_status(bp, 120, 139)
            chol_status, chol_color = get_status(cholesterol, 199, 239)
            bs_status, bs_color = ("Normal", "green") if bs == 0 else ("High Risk", "red")
            
            hr_status, hr_color = "Normal", "green"
            if hr < 60 or hr > 100:
                hr_status, hr_color = ("Slightly High" if hr <= 120 else "High Risk"), "red"
            
            comparisons = {
                'bp': {'ideal': '≤ 120', 'your': bp, 'status': bp_status, 'color': bp_color},
                'cholesterol': {'ideal': '< 200', 'your': cholesterol, 'status': chol_status, 'color': chol_color},
                'bs': {'ideal': 'No (>120)', 'your': 'Yes' if bs == 1 else 'No', 'status': bs_status, 'color': bs_color},
                'hr': {'ideal': '60-100', 'your': hr, 'status': hr_status, 'color': hr_color}
            }
            
            
            return render_template('result.html', 
                                   risk_level=risk_level, 
                                   confidence=round(confidence, 1),
                                   color_theme=color_theme,
                                   recommendation=recommendation,
                                   comparisons=comparisons,
                                   user=session['user'])
                                   
        except Exception as e:
            return f"An error occurred: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
