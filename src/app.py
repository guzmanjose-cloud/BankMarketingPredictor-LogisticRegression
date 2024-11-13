from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('/workspaces/BankMarketingPredictor-LogisticRegression/src/logistic_regression_model.pkl')
scaler = joblib.load('/workspaces/BankMarketingPredictor-LogisticRegression/src/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve values from form
        age = float(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        contact = request.form['contact']
        month = request.form['month']
        poutcome = request.form['poutcome']  # New input for poutcome
        economic_pca = float(request.form['economic_pca'])
        campaign_pca = float(request.form['campaign_pca'])

        # Encode categorical features based on the training data (adjust if necessary)
        job_mapping = {
            "admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3,
            "management": 4, "retired": 5, "self-employed": 6, "services": 7,
            "student": 8, "technician": 9, "unemployed": 10, "unknown": 11
        }
        marital_mapping = {"single": 0, "married": 1, "divorced": 2}
        education_mapping = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3}
        contact_mapping = {"cellular": 0, "telephone": 1, "unknown": 2}
        month_mapping = {
            "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
            "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
        }
        poutcome_mapping = {"success": 0, "failure": 1, "other": 2, "unknown": 3}  # New mapping for poutcome

        # Map the categorical features
        job_encoded = job_mapping.get(job, -1)
        marital_encoded = marital_mapping.get(marital, -1)
        education_encoded = education_mapping.get(education, -1)
        contact_encoded = contact_mapping.get(contact, -1)
        month_encoded = month_mapping.get(month, -1)
        poutcome_encoded = poutcome_mapping.get(poutcome, -1)  # Encode poutcome

        # Collect all feature values into a single array, including poutcome
        features = np.array([
            age, job_encoded, marital_encoded, education_encoded, contact_encoded,
            month_encoded, poutcome_encoded, economic_pca, campaign_pca
        ]).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        prediction_label = "yes" if prediction == 1 else "no"


        # Render the HTML with prediction result
        return render_template('index.html', prediction=(prediction_label))
    except Exception as e:
        # Log or print the error for debugging
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction="Error processing request.")

if __name__ == '__main__':
    app.run(debug=True)
