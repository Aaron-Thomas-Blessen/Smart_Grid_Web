# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Load model and scalers
model = tf.keras.models.load_model('path_to_your_saved_model.h5')  # Replace with your model path
scaler_load = MinMaxScaler(feature_range=(0, 1))
scaler_city = MinMaxScaler(feature_range=(0, 1))
# Load scaler and city data if saved separately
# Ensure you have your model and scalers saved correctly for deployment

# Function for load forecasting
def predict_future_load(input_date, city_name):
    input_date = pd.to_datetime(input_date)
    city_name = city_name.strip().lower()
    
    # Ensure valid city name
    if city_name not in city_encoded.columns:
        raise ValueError("City name not found. Available cities: {}".format(', '.join(city_encoded.columns)))

    future_sequence = scaled_data[-sequence_length:, :].copy()
    city_feature_index = list(city_encoded.columns).index(city_name)
    city_feature_values = np.zeros(len(city_encoded.columns))
    city_feature_values[city_feature_index] = 1
    city_feature_values_scaled = scaler_city.transform(pd.DataFrame([city_feature_values], columns=city_encoded.columns))
    
    days_to_predict = (input_date - energy_data.index[-1]).days
    if days_to_predict <= 0:
        raise ValueError("Input date must be in the future.")

    for _ in range(days_to_predict):
        predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))
        city_feature_values_scaled = np.tile(city_feature_values_scaled, (predicted_scaled.shape[0], 1))
        new_step = np.hstack((predicted_scaled, city_feature_values_scaled))
        future_sequence = np.vstack((future_sequence[1:], new_step))

    predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))
    return predicted_load[0, 0]

# Streamlit UI
st.title("Smart Grid Load Forecasting")

# Input Section
st.sidebar.header("Enter Details")
input_date = st.sidebar.date_input("Select a future date")
city_name = st.sidebar.selectbox("Select City", city_encoded.columns)

# Prediction Button
if st.sidebar.button("Predict Load"):
    try:
        # Call prediction function
        future_load = predict_future_load(input_date, city_name)
        st.subheader(f"Predicted Load for {city_name.capitalize()} on {input_date}:")
        st.write(f"{future_load:.2f} MW")
    except ValueError as e:
        st.error(e)

# Display R-squared Accuracy
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
accuracy_percentage = r2 * 100
st.sidebar.write(f'Accuracy (R-squared): {accuracy_percentage:.2f}%')
