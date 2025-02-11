import streamlit as st
import numpy as np
import pickle  # Assuming the RandomForest model and scaler are saved as pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained RandomForest model and scaler (replace with your actual paths)
rf_clf = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Function to preprocess the input based on training preprocessing
def preprocess_input(data):
    # Label encoding for categorical variables
    label_encoder = {
        'PreferredLoginDevice': LabelEncoder().fit(['Mobile Phone', 'Computer']),
        'PreferredPaymentMode': LabelEncoder().fit(['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet']),
        'Gender': LabelEncoder().fit(['Female', 'Male']),
        'PreferedOrderCat': LabelEncoder().fit(['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery']),
        'MaritalStatus': LabelEncoder().fit(['Single', 'Divorced', 'Married']),
        'Complain': LabelEncoder().fit([0, 1])
    }
    
    # Apply label encoding
    for feature in label_encoder.keys():
        data[feature] = label_encoder[feature].transform([data[feature]])[0]
    
    # Convert to DataFrame
    input_data = np.array([[
        data['Tenure'], data['PreferredLoginDevice'], data['CityTier'], data['WarehouseToHome'],
        data['PreferredPaymentMode'], data['Gender'], data['HourSpendOnApp'], data['NumberOfDeviceRegistered'],
        data['PreferedOrderCat'], data['SatisfactionScore'], data['MaritalStatus'], data['NumberOfAddress'],
        data['Complain'], data['OrderAmountHikeFromlastYear'], data['CouponUsed'], data['OrderCount'],
        data['DaySinceLastOrder'], data['CashbackAmount']
    ]])

    # Apply scaling
    input_data = scaler.transform(input_data)
    
    return input_data

# Streamlit app
st.title("🌟 Customer Churn Prediction App 🌟")
st.sidebar.header("Input Features")

# Function to get user input
def user_input_features():
    tenure = st.sidebar.number_input("Tenure (months) 🗓️", min_value=0.0, step=1.0)
    preferred_login_device = st.sidebar.selectbox("Preferred Login Device 📱", ['Mobile Phone', 'Computer'])
    city_tier = st.sidebar.selectbox("City Tier 🌆", [1, 2, 3])
    warehouse_to_home = st.sidebar.number_input("Warehouse To Home Distance (km) 🚚", min_value=0.0, step=1.0)
    preferred_payment_mode = st.sidebar.selectbox("Preferred Payment Mode 💳", ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])
    gender = st.sidebar.selectbox("Gender 🚻", ['Female', 'Male'])
    hour_spend_on_app = st.sidebar.number_input("Hour Spend On App (hours) ⏰", min_value=0.0, max_value=None, step=1.0)
    number_of_device_registered = st.sidebar.number_input("Number of Devices Registered 💻", min_value=1, max_value=15, step=1)
    prefered_order_cat = st.sidebar.selectbox("Preferred Order Category 🛍️", ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
    satisfaction_score = st.sidebar.selectbox("Satisfaction Score (1-5) ⭐", [1, 2, 3, 4, 5])
    marital_status = st.sidebar.selectbox("Marital Status 💍", ['Single', 'Divorced', 'Married'])
    number_of_address = st.sidebar.number_input("Number of Addresses 🏡", min_value=1, max_value=22, step=1)
    complain = st.sidebar.selectbox("Complain (0=No, 1=Yes) 🚨", [0, 1])
    order_amount_hike_from_last_year = st.sidebar.number_input("Order Amount Hike From Last Year ($) 📈", min_value=0.0, step=1.0)
    coupon_used = st.sidebar.number_input("Coupon Used ($) 🏷️", min_value=0.0, step=1.0)
    order_count = st.sidebar.number_input("Order Count 📦", min_value=0.0, step=1.0)
    day_since_last_order = st.sidebar.number_input("Days Since Last Order 📅", min_value=0.0, step=1.0)
    cashback_amount = st.sidebar.number_input("Cashback Amount ($) 💵", min_value=0.0, step=1.0)

    return {
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device_registered,
        'PreferedOrderCat': prefered_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': complain,
        'OrderAmountHikeFromlastYear': order_amount_hike_from_last_year,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount
    }

# Initialize history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Collect all the input data
input_data = user_input_features()

# When the user clicks on Predict
if st.button('Predict 🤔'):
    # Preprocess the data
    processed_data = preprocess_input(input_data)

    # Make predictions and get probabilities
    predictions = rf_clf.predict(processed_data)
    predicted_proba = rf_clf.predict_proba(processed_data)

    # Prepare result data
    result = {
        'input': input_data,
        'predicted_churn': 'Yes' if predictions[0] == 1 else 'No',
        'prediction_probabilities': predicted_proba[0].tolist()  # Convert to list for easier display
    }
    
    # Add the result to the history
    st.session_state.history.append(result)

    # Display the prediction and its probability
    st.subheader("Prediction Results 🎯")
    st.write(f"**Predicted Churn:** {'Yes' if predictions[0] == 1 else 'No'}")

    # Define class labels
    class_labels = ["No Churn", "Churn"]  # Modify as per your classification labels

    # Display prediction probabilities with class labels
    st.write("**Prediction Probabilities:**")
    for i, prob in enumerate(predicted_proba[0]):
        st.write(f"**{class_labels[i]}:** {prob:.2f}")

    # Display history of predictions
    st.subheader("Prediction History 📜")
    for i, res in enumerate(st.session_state.history):
        st.write(f"**Prediction {i + 1}:** {res['predicted_churn']} | Input: {res['input']} | Probabilities: {res['prediction_probabilities']}")
