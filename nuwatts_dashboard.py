# Nuwatts AI Model Training Notebook with Live Dashboard

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import time

# Step 2: Load and Preprocess Data
df = pd.read_csv("Nuwatts_AI_Training_Data_61_Minutes.csv")
time_column = pd.read_csv("Nuwatts_AI_Training_Data_61_Minutes.csv")[["Time"]]  # Keep original time for charting

# Step 3: Normalize the Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop(columns=["Time"]))
scaled_df = pd.DataFrame(scaled_data, columns=df.columns.drop("Time"))

# Step 4: Split Features and Target
X = scaled_df.drop(columns=["TEG Power (W)"])
y = scaled_df["TEG Power (W)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 7: Feature Importance
importances = model.feature_importances_
features = X.columns

# Streamlit Dashboard
st.title("Nuwatts TEG Power Prediction Dashboard")
st.markdown(f"**Model Performance:** R² = {r2:.2f}, MAE = {mae:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features, importances)
ax.set_xlabel("Feature Importance")
ax.set_title("What Factors Most Affect TEG Power Output")
st.pyplot(fig)

# Step 8: Real-Time Inference Simulation with Live Chart
st.subheader("Live Inference Simulation")
metric_placeholder = st.empty()
chart_placeholder = st.empty()

predicted_values = []
plot_data = pd.DataFrame(columns=["Minute", "Predicted TEG Power (normalized)", "Ferrofluid Temp (°C)", "Magnetic Field Strength (kA/m)", "Heat Exchanger Temp (°C)"])

for i in range(len(X)):
    input_data = X.iloc[i:i+1]
    predicted_power = model.predict(input_data)[0]
    predicted_values.append(predicted_power)

    # Update metric
    metric_placeholder.metric(label=f"Minute {i+1}", value=f"{predicted_power:.4f} (normalized)")

    # Append corresponding real-world features for charting
    real_data_point = df.iloc[i]
    plot_data = plot_data.append({
        "Minute": i + 1,
        "Predicted TEG Power (normalized)": predicted_power,
        "Ferrofluid Temp (°C)": real_data_point["Ferrofluid Temp (°C)"],
        "Magnetic Field Strength (kA/m)": real_data_point["Magnetic Field Strength (kA/m)"],
        "Heat Exchanger Temp (°C)": real_data_point["Heat Exchanger Temp (°C)"]
    }, ignore_index=True)

    # Display updated chart with selected features
    chart_placeholder.line_chart(plot_data.set_index("Minute"))

    time.sleep(0.1)
