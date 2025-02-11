import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained RandomForest model and scaler
rf_clf = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Function to preprocess the input based on training preprocessing
def preprocess_input(data):
    label_encoder = {
        'PreferredLoginDevice': LabelEncoder().fit(['Mobile Phone', 'Computer']),
        'PreferredPaymentMode': LabelEncoder().fit(['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet']),
        'Gender': LabelEncoder().fit(['Female', 'Male']),
        'PreferedOrderCat': LabelEncoder().fit(['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery']),
        'MaritalStatus': LabelEncoder().fit(['Single', 'Divorced', 'Married']),
        'Complain': LabelEncoder().fit([0, 1])
    }
    
    # Apply label encoding
    data['PreferredLoginDevice'] = label_encoder['PreferredLoginDevice'].transform([data['PreferredLoginDevice']])[0]
    data['PreferredPaymentMode'] = label_encoder['PreferredPaymentMode'].transform([data['PreferredPaymentMode']])[0]
    data['Gender'] = label_encoder['Gender'].transform([data['Gender']])[0]
    data['PreferedOrderCat'] = label_encoder['PreferedOrderCat'].transform([data['PreferedOrderCat']])[0]
    data['MaritalStatus'] = label_encoder['MaritalStatus'].transform([data['MaritalStatus']])[0]
    data['Complain'] = label_encoder['Complain'].transform([data['Complain']])[0]
    
    input_data = np.array([[
        data['Tenure'], data['PreferredLoginDevice'], data['CityTier'], data['WarehouseToHome'],
        data['PreferredPaymentMode'], data['Gender'], data['HourSpendOnApp'], data['NumberOfDeviceRegistered'],
        data['PreferedOrderCat'], data['SatisfactionScore'], data['MaritalStatus'], data['NumberOfAddress'],
        data['Complain'], data['OrderAmountHikeFromlastYear'], data['CouponUsed'], data['OrderCount'],
        data['DaySinceLastOrder'], data['CashbackAmount']
    ]])

    input_data = scaler.transform(input_data)
    
    return input_data

# Function to run predictions on a given logfile
def run_predictions(logfile):
    # Load the data
    data = pd.read_csv(logfile)
    
    # Check if 'Churn' column exists and drop it
    if 'Churn' in data.columns:
        data = data.drop(columns=['Churn'])

    # Ensure the input data is structured correctly
    required_columns = [
        'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
        'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 
        'NumberOfDeviceRegistered', 'PreferedOrderCat', 
        'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
        'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
    ]
    
    # Ensure the dataframe has the required columns and in the correct order
    data = data[required_columns]
    
    # Preprocess the data for prediction
    processed_data = scaler.transform(data)  # Assuming 'scaler' is already fitted

    # Make predictions
    predictions = rf_clf.predict(processed_data)
    return predictions

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("Customer Churn Prediction Testing")

# Add a sidebar for inputs
st.sidebar.header("Run Predictions")
st.sidebar.markdown("Select a logfile to run predictions:")

# Buttons to run predictions for the two log files
if st.sidebar.button('Run Predictions for logfile_1 (Churned Customers)'):
    predictions_logfile_1 = run_predictions('logfile_1.csv')
    count_not_churned = np.sum(predictions_logfile_1 != 1)
    st.success(f"Predictions for logfile_1: {predictions_logfile_1.tolist()}")
    st.write(f"Number of predictions not equal to 1: {count_not_churned}")

if st.sidebar.button('Run Predictions for logfile_0 (Non-Churned Customers)'):
    predictions_logfile_0 = run_predictions('logfile_0.csv')
    count_not_non_churned = np.sum(predictions_logfile_0 != 0)
    st.success(f"Predictions for logfile_0: {predictions_logfile_0.tolist()}")
    st.write(f"Number of predictions not equal to 0: {count_not_non_churned}")

# Add any other components you want (like the input fields you already have) here.

# Optional: Add some footer or additional information
st.markdown("---")
st.markdown("### About This App")
st.write("This application predicts customer churn based on various features. Upload your data and get instant predictions!")
