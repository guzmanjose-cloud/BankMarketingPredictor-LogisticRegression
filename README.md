Bank Marketing Campaign Prediction

This project is a Flask-based web application that predicts customer responses to a bank's marketing campaign using a logistic regression model. The model uses features like customer demographics, previous interactions, and economic indicators to predict whether a customer will respond positively to the campaign.
Table of Contents

    Overview
    Dataset
    Features
    Project Structure
    Installation
    Usage
    Model Details
    Future Improvements

Overview

The goal of this project is to build a machine learning model that predicts customer responses to a bank’s term deposit marketing campaign. The model and app can help banks target their campaigns more effectively by identifying which customers are most likely to respond positively to specific marketing strategies.
Dataset

The dataset used in this project comes from a real-world bank marketing campaign and contains information on customer demographics, previous campaign contacts, economic indicators, and campaign-specific details. The target variable, y, indicates whether the customer purchased the bank's product (e.g., a term deposit) after the campaign.

    Target Variable (y): "yes" (customer purchased the product) or "no" (customer did not purchase the product).

Features

The dataset includes various features grouped as follows:

    Customer Demographics:
        age, job, marital, education, etc.

    Economic Indicators:
        employment_rate_variation, euribor_3_month, number_employed
        These were reduced to a single economic_pca component using Principal Component Analysis (PCA).

    Campaign-Specific Features:
        duration, campaign, previous
        These were reduced to a single campaign_pca component using PCA.

Project Structure

BankMarketingPredictor
├── src
│   ├── app.py               # Flask application
│   ├── logistic_regression_model.pkl  # Trained logistic regression model
│   ├── scaler.pkl           # Pre-fitted StandardScaler for data scaling
│   └── templates
│       └── index.html       # HTML form for user input
├── data
│   └── bank_marketing_campaign_data.csv  # Dataset
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

Installation

    Clone the Repository:

git clone https://github.com/yourusername/BankMarketingPredictor.git
cd BankMarketingPredictor

Install Dependencies: Make sure to create a virtual environment, then install the necessary dependencies:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    Set Up Files: Place logistic_regression_model.pkl and scaler.pkl in the src directory.

Usage

    Run the Flask App: Start the Flask app from the src directory:

    python app.py

    Access the Web Interface: Open your web browser and go to http://127.0.0.1:5000. You will see a form where you can input customer information and campaign details.

    Get Predictions: After filling in the form, submit it to get a prediction on whether the customer is likely to respond positively to the campaign.

Model Details

The logistic regression model was trained to predict the likelihood of a customer accepting a bank's product offer based on:

    Customer demographics
    Economic indicators
    Previous and current campaign details

The model was trained with features scaled by StandardScaler and reduced in complexity by applying PCA to economic and campaign-related features. This preprocessing improves model performance and interpretability.
Feature Engineering

    Economic PCA Component: Summarizes economic indicators, indicating overall economic conditions.
    Campaign PCA Component: Summarizes the intensity of campaign interactions with the customer.

Performance

    The model was evaluated on accuracy, precision, recall, and F1-score. (Add specific results if available)

Future Improvements

    Model Tuning: Experiment with other algorithms, such as decision trees or ensemble methods.
    Feature Engineering: Explore additional economic or behavioral features to enhance predictive power.
    UI Improvements: Enhance the web interface with more detailed explanations or interactive visualizations.
