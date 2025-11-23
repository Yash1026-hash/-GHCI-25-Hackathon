from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from transaction_classifier_api import APICategorizer
import os
import random
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'demo_secret_key_123'  # Required for session

# Initialize categorizer
categorizer = APICategorizer(
    taxonomy_path='taxonomy.yaml',
    api_provider='web_lookup'
)

# Mock Data for Simulation
MOCK_HISTORY = [
    {"merchant": "Starbucks", "amount": 5.50, "date": "2023-10-25"},
    {"merchant": "Shell Station", "amount": 45.00, "date": "2023-10-24"},
    {"merchant": "Netflix", "amount": 15.99, "date": "2023-10-23"},
    {"merchant": "Whole Foods", "amount": 124.50, "date": "2023-10-22"},
    {"merchant": "Uber Trip", "amount": 24.30, "date": "2023-10-21"},
    {"merchant": "Spotify", "amount": 9.99, "date": "2023-10-20"},
    {"merchant": "Amazon", "amount": 34.99, "date": "2023-10-19"},
    {"merchant": "McDonalds", "amount": 12.40, "date": "2023-10-18"},
    {"merchant": "Electric Bill", "amount": 85.00, "date": "2023-10-15"},
    {"merchant": "Target", "amount": 55.20, "date": "2023-10-14"}
]

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Demo login - accept any non-empty input
        username = request.form.get('username')
        if username:
            session['user'] = username
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/api/categorize', methods=['POST'])
def categorize():
    data = request.json
    raw_string = data.get('raw_string', '')
    amount = float(data.get('amount', 0))
    
    result = categorizer.categorize(raw_string, amount)
    return jsonify(result)

@app.route('/api/explain', methods=['POST'])
def explain():
    data = request.json
    raw_string = data.get('raw_string', '')
    amount = float(data.get('amount', 0))
    
    explanation = categorizer.explain_prediction(raw_string, amount)
    return jsonify(explanation)

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    transaction_id = data.get('transaction_id', 'unknown')
    raw_string = data.get('raw_string', '')
    correct_category = data.get('correct_category', '')
    
    categorizer.log_feedback(transaction_id, raw_string, correct_category)
    return jsonify({'status': 'success', 'message': 'Feedback logged'})

@app.route('/api/connect_wallet', methods=['POST'])
def connect_wallet():
    # Simulate API delay
    import time
    time.sleep(1.5)
    return jsonify({'status': 'success', 'message': 'Connected to PhonePe'})

@app.route('/api/history', methods=['GET'])
def get_history():
    # Return mock history with categories
    processed_history = []
    for txn in MOCK_HISTORY:
        cat_result = categorizer.categorize(txn['merchant'], txn['amount'])
        processed_history.append({
            'merchant': txn['merchant'],
            'amount': txn['amount'],
            'date': txn['date'],
            'category': cat_result['category'],
            'confidence': cat_result['confidence']
        })
    return jsonify(processed_history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
