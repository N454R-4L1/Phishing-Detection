from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained models
with open('tree_model.pkl', 'rb') as f:
    tree_model = pickle.load(f)
with open('forest_model.pkl', 'rb') as f:
    forest_model = pickle.load(f)

def extract_features(url):
    # This example shows 16 different features.
    # Adjust as necessary based on what was used during training.
    features = [
        len(url),                         # Length of URL
        int('https' in url),              # HTTPS presence (1 if true, 0 otherwise)
        url.count('.'),                   # Number of dots
        url.count('/'),                   # Number of slashes
        url.count('-'),                   # Number of hyphens
        url.count('?'),                   # Number of question marks
        int(url.startswith('http')),      # HTTP presence
        int(url.startswith('https')),     # HTTPS presence at the start
        len(url.split('/')[0]),           # Length of the domain name
        int('.com' in url),               # Presence of .com
        int('.net' in url),               # Presence of .net
        int('.org' in url),               # Presence of .org
        int('login' in url.lower()),      # Contains "login"
        int('admin' in url.lower()),      # Contains "admin"
        int('@' in url),                  # Presence of '@' symbol
        int('=' in url),                  # Presence of '=' symbol
    ]
    return np.array(features).reshape(1, -1)  # Reshape for sklearn


# Route for serving index.html
@app.route('/')
def home():
    return render_template('index.html')

# Classification route
@app.route('/classify', methods=['POST'])
def classify_url():
    data = request.json
    url = data.get('url')
    if url:
        features = extract_features(url)
        tree_pred = tree_model.predict(features)[0]
        forest_pred = forest_model.predict(features)[0]
        result = 'phishing' if tree_pred == 1 or forest_pred == 1 else 'legitimate'
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No URL provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
