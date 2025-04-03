import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Load dataset
def load_data():
    df = pd.read_csv("/content/sample_data/water_potability.csv")  # Update this to the correct file path
    return df

# Preprocess data
def preprocess_data(df):
    # Handle missing values (you can use different strategies)
    df.fillna(df.mean(), inplace=True)

    # Normalize the numerical features
    numerical_features = df.drop(columns=["Potability"])  # Excluding the target column (Potability)
    scaler = MinMaxScaler()
    df[numerical_features.columns] = scaler.fit_transform(numerical_features)

    # Encoding Potability (0 or 1)
    y = df["Potability"].values
    return df, y, scaler

# Create sequences for LSTM
def create_sequences(data, target, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(target[i + seq_length])  # Predicting potability (0 or 1)
    return np.array(sequences), np.array(labels)

# Create LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1, activation='sigmoid')  # Predicting Potability (binary classification)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(X_train, y_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    return model

# Streamlit UI
def main():
    st.title("Water Potability Prediction using LSTM")

    # Upload the dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Show the dataframe and check for missing values
        st.write("Dataset Overview", df.head())
        st.write("Missing values per column", df.isnull().sum())

        # Preprocess the data
        df, y, scaler = preprocess_data(df)

        # Create sequences (for LSTM)
        seq_length = 5
        X, y = create_sequences(df.values, y, seq_length)

        # Split data into training and testing sets
        X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
        y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)
        st.write(f"Test accuracy: {accuracy[1]:.4f}")

        # Make predictions
        predictions = model.predict(X_test)
        predicted_labels = (predictions > 0.5).astype(int)

        # Display the first 10 predictions
        st.write("First 10 Predictions vs Actual Potability")
        results = pd.DataFrame({
            'Predicted': predicted_labels.flatten(),
            'Actual': y_test[:10]
        })
        st.write(results)

        # Visualizing the results
        st.subheader("Water Potability Distribution")
        fig = px.histogram(results, x="Predicted", color="Actual", title="Predicted vs Actual Potability")
        st.plotly_chart(fig)

        # Show the line chart of prediction trends (for a quick visual)
        st.subheader("Prediction Trend for the First 100 Data Points")
        st.line_chart(predictions[:100].flatten())

if __name__ == "__main__":
    main()
