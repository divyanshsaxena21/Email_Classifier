# app.py
from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

# Load saved components
model = joblib.load('xgb_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    tokens = [w for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    subject = data.get('subject', '')
    body = data.get('body', '')

    clean_subject = clean_text(subject)
    clean_body = clean_text(body)
    combined_text = clean_subject + ' ' + clean_body

    X_input = vectorizer.transform([combined_text])
    y_pred = model.predict(X_input)
    label = label_encoder.inverse_transform(y_pred)[0]

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
