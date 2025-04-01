import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import plotly.express as px

# Load dataset
def load_data():
    df = pd.read_csv("ganga_water_quality.csv")  # Ensure the correct path
    return df

# Preprocess data
def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)

    # Encode Station-Location
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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
    if len(data) < seq_length + 1:
        raise ValueError("Not enough data to create sequences. Try reducing seq_length.")
    
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, -2:])
    return np.array(sequences), np.array(labels)


# Create LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(2)  # Predicting DO & BOD
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(X_train, y_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    return model

# Streamlit UI
def main():
    st.set_page_config(page_title="Ganga Water Quality Forecast", layout="wide")
    st.title("🌊 Ganga River Water Quality Forecasting")

    df = load_data()
    df, scaler = preprocess_data(df)

    # Sidebar filter
    station_list = df.columns[:-5]  # Excluding numerical columns
    selected_station = st.sidebar.selectbox("Select a Station", station_list)
    
    # Show data summary
    st.sidebar.subheader("Dataset Overview")
    st.sidebar.write(df.describe())

    # Historical Trends
    st.subheader("📊 Historical Water Quality Data")
    fig1 = px.line(df, x=df.index, y=['Dissolved Oxygen during 2011 (mg/l)', 'Biological Oxygen demand during 2011 (mg/l)'],
                   title="Dissolved Oxygen & BOD Over Time")
    st.plotly_chart(fig1)

    seq_length = 10
    X, y = create_sequences(df.values, seq_length)
    X_train, y_train = X[:-10], y[:-10]
    st.write(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"📌 X_train first sample:\n{X_train[:1]}")
    st.write(f"📌 y_train first sample:\n{y_train[:1]}")

    model = train_model(X_train, y_train)

    future_predictions = model.predict(X[-10:])

    # Placeholder array for inverse transformation
    num_features = df.shape[1]
    dummy = np.zeros((future_predictions.shape[0], num_features))
    dummy[:, -2:] = future_predictions
    future_predictions = scaler.inverse_transform(dummy)[:, -2:]

    pred_df = pd.DataFrame({
        'Time': range(len(future_predictions)),
        'Dissolved Oxygen': future_predictions[:, 0],
        'Biological Oxygen Demand': future_predictions[:, 1]
    })

    # Forecast Chart
    st.subheader("🔮 Future Predictions")
    fig2 = px.line(pred_df, x='Time', y=['Dissolved Oxygen', 'Biological Oxygen Demand'],
                   title="Forecasted DO & BOD")
    st.plotly_chart(fig2)

if __name__ == "__main__":
    main()
