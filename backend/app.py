from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import numpy as np
import shap
import json
import os

app = Flask(__name__, static_folder='../frontend')

# Load the model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Optional: Initialize SHAP explainer for the backend
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']
if type(classifier).__name__ in ['RandomForestClassifier', 'XGBClassifier']:
    explainer = shap.TreeExplainer(classifier)
else:
    explainer = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Reconstruct a DataFrame for a single row
    # The frontend should send exactly the fields expected by the preprocessor
    df = pd.DataFrame([data])
    
    # Model prediction
    prob = model.predict_proba(df)[0][1]
    
    # Hardcode best threshold based on evaluation
    threshold = 0.4
    classification = "High Risk" if prob >= threshold else "Low Risk"
    
    # Generate local SHAP explanation
    explanation = []
    if explainer is not None:
        X_transformed = preprocessor.transform(df)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]
            
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        all_features = list(num_features) + list(cat_features)
        
        # Get top 3 factors for this specific prediction
        feature_contributions = list(zip(all_features, sv))
        # Sort by absolute contribution, but keep the sign
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feat, val in feature_contributions[:3]:
            direction = "increased" if val > 0 else "decreased"
            explanation.append({
                "feature": feat,
                "contribution": round(float(val), 4),
                "direction": direction
            })
            
    return jsonify({
        "probability": round(float(prob), 4),
        "classification": classification,
        "threshold": threshold,
        "explanation": explanation
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
