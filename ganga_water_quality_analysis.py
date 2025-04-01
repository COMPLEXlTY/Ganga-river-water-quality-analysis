# -*- coding: utf-8 -*-
"""ganga_water_quality_analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/105BzHJN9_lghKPsha7WM9ytTXXWdCMty
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load dataset
def load_data():
    df = pd.read_csv("ganga_water_quality.csv")
    return df

# Preprocess data
def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)

    # Encode Station-Location
    encoder = OneHotEncoder(sparse_output =False, handle_unknown='ignore')
    station_encoded = encoder.fit_transform(df[['Station-Location']])
    station_df = pd.DataFrame(station_encoded, columns=encoder.get_feature_names_out())

    # Normalize numerical data
    scaler = MinMaxScaler()
    numerical_features = ['Distance in Kms.', 'Dissolved Oxygen during 1986 (mg/l)',
                          'Biological Oxygen Demand in 1986 (mg/l)',
                          'Dissolved Oxygen during 2011 (mg/l)', 'Biological Oxygen demand during 2011 (mg/l)']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    df = pd.concat([station_df, df[numerical_features]], axis=1)
    return df, scaler

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, -2:])
    return np.array(sequences), np.array(labels)

# Create LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(2)  # Predicting DO & BOD
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(X_train, y_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

# Streamlit UI
def main():
    st.title("Ganga River Water Quality Forecasting")
    df = load_data()
    df, scaler = preprocess_data(df)
    seq_length = 5
    X, y = create_sequences(df.values, seq_length)
    X_train, y_train = X[:-5], y[:-5]
    model = train_model(X_train, y_train)

    future_predictions = model.predict(X[-5:])
    # Use the same numerical features as in preprocess_data()
    numerical_features = ['Distance in Kms.', 'Dissolved Oxygen during 1986 (mg/l)',
                      'Biological Oxygen Demand in 1986 (mg/l)',
                      'Dissolved Oxygen during 2011 (mg/l)', 'Biological Oxygen demand during 2011 (mg/l)']

# Get the number of numerical features
    num_features = len(numerical_features)

# Create a placeholder array with the correct shape
    dummy = np.zeros((future_predictions.shape[0], num_features))

# Assign predictions to the last two columns (DO & BOD)
    dummy[:, -2:] = future_predictions

# Apply inverse transform and extract only DO & BOD
    future_predictions = scaler.inverse_transform(dummy)[:, -2:]

    # Create a DataFrame for better visualization
    pred_df = pd.DataFrame({
        'Dissolved Oxygen': future_predictions[:, 0],
        'Biological Oxygen Demand': future_predictions[:, 1]
    })

# Display the line chart
    st.line_chart(pred_df)


if __name__ == "__main__":
    main()

