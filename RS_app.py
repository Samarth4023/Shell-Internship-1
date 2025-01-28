import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the saved Blend Ensemble model
ensemble_model_loaded = joblib.load("Models/Crop_Recommendation_model.pkl")

# Extract base models and meta-model
models_loaded = ensemble_model_loaded['base_models']
meta_model_loaded = ensemble_model_loaded['meta_model']

# Feature names (ensure these match the training dataset)
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Streamlit UI
st.title("Crop and Fertilizer Recommendation System")

# Crop Recommendation
st.header("Crop Recommendation")
st.write("Enter the details:")

# Input fields for new data (user input) with unique keys
N = st.number_input('Enter Nitrogen (N)', min_value=0, max_value=300, step=1, key="N")
P = st.number_input('Enter Phosphorous (P)', min_value=0, max_value=300, step=1, key="P")
K = st.number_input('Enter Potassium (K)', min_value=0, max_value=300, step=1, key="K")
temperature = st.number_input('Enter Temperature (°C)', min_value=10, max_value=50, step=1, key="temperature")
humidity = st.number_input('Enter Humidity (%)', min_value=0, max_value=100, step=1, key="humidity")
ph = st.number_input('Enter pH', min_value=4.0, max_value=9.0, step=0.1, key="ph")
rainfall = st.number_input('Enter Rainfall (mm)', min_value=0, max_value=1000, step=1, key="rainfall")

# Button to trigger prediction
if st.button("Get Crop Recommendation"):
    new_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

    # Get the predicted probabilities from each base model
    base_predictions_new = []
    for model in models_loaded.values():
        base_pred_new = model.predict_proba(new_data)  # Get probabilities for each class
        base_predictions_new.append(base_pred_new)

    # Stack the predictions from all base models
    stacked_predictions_new = np.hstack(base_predictions_new)

    # Get final prediction using the meta-model
    final_pred_new = meta_model_loaded.predict(stacked_predictions_new)

    # Output the final prediction
    st.write(f"Prediction for the entered data: {final_pred_new[0]}")

    # Show model performance metrics
    st.subheader("Model Performance Metrics:")
    accuracy = meta_model_loaded.score(stacked_predictions_new, final_pred_new)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    #st.write(models_loaded.keys())

    # Assuming the base model has feature importance (e.g., RandomForestClassifier)
    crop_feature_importance = models_loaded['RandomForestClassifier'].feature_importances_

    # Create a bar chart
    st.subheader("Feature Importance:")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names, crop_feature_importance)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Create a DataFrame from the input data
    st.subheader("Feature Distributions:")
    new_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    new_data.plot(kind='bar', ax=ax)
    ax.set_title('Feature Distribution')
    st.pyplot(fig)

    # Plot probabilities
    st.subheader("Prediction Probabilities:")
    predicted_class = final_pred_new[0]
    class_probabilities = stacked_predictions_new[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(class_probabilities)), class_probabilities)
    ax.set_title(f'Prediction Probabilities for {predicted_class}')
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Probability')
    st.pyplot(fig)

    # Add prediction to the DataFrame
    new_data["prediction"] = final_pred_new

    # Download results as CSV
    st.download_button(label="Download Crop Recommendation Results", data=new_data.to_csv(), file_name="Crop Recommendation Results.csv", mime="text/csv")

# Fertilizer Recommendation
st.header("Fertilizer Recommendation System")
st.write("Enter the details:")

# Load the saved Blend Ensemble model for fertilizer recommendation
fertilizer_ensemble_model = joblib.load("Models/Fertilizer_Recommendation_model.pkl")

# Extract base models and meta-model
fertilizer_base_models = fertilizer_ensemble_model['base_models1']
fertilizer_meta_model = fertilizer_ensemble_model['meta_model1']

# Feature names (ensure these match the training dataset for fertilizer recommendation)
fertilizer_feature_names = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create LabelEncoders for categorical columns
soil_type_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()

# Fit encoders with the training data categories (ensure these match the training process)
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']  # Replace with actual soil types in training
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']  # Replace with actual crop types in training
soil_type_encoder.fit(soil_types)
crop_type_encoder.fit(crop_types)

# Input fields for new data (user input) with unique keys
temperature_fertilizer = st.number_input('Enter Temperature (°C)', min_value=0, max_value=50, step=1, key="fertilizer_temperature")
humidity_fertilizer = st.number_input('Enter Humidity (%)', min_value=0, max_value=100, step=1, key="fertilizer_humidity")
moisture = st.number_input('Enter Moisture (%)', min_value=0, max_value=100, step=1, key="fertilizer_moisture")
soil_type = st.selectbox('Select Soil Type', soil_types, key="fertilizer_soil_type")
crop_type = st.selectbox('Select Crop Type', crop_types, key="fertilizer_crop_type")
nitrogen = st.number_input('Enter Nitrogen (N)', min_value=0, max_value=300, step=1, key="fertilizer_nitrogen")
potassium = st.number_input('Enter Potassium (K)', min_value=0, max_value=300, step=1, key="fertilizer_potassium")
phosphorous = st.number_input('Enter Phosphorous (P)', min_value=0, max_value=300, step=1, key="fertilizer_phosphorous")

# Button to trigger prediction
if st.button("Get Fertilizer Recommendation"):
    # Prepare input data
    fertilizer_new_data = pd.DataFrame([[temperature_fertilizer, humidity_fertilizer, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]], columns=fertilizer_feature_names)

    # Encode categorical features
    fertilizer_new_data['Soil Type'] = soil_type_encoder.transform(fertilizer_new_data['Soil Type'])
    fertilizer_new_data['Crop Type'] = crop_type_encoder.transform(fertilizer_new_data['Crop Type'])

    # Convert to NumPy array for model compatibility
    fertilizer_new_data = fertilizer_new_data.to_numpy()

    # Get the predicted probabilities from each base model
    fertilizer_base_predictions_new = []
    for model in fertilizer_base_models.values():
        base_pred_new = model.predict_proba(fertilizer_new_data)  # Get probabilities for each class
        fertilizer_base_predictions_new.append(base_pred_new)

    # Stack the predictions from all base models
    fertilizer_stacked_predictions_new = np.hstack(fertilizer_base_predictions_new)

    # Get the final prediction using the meta-model
    fertilizer_final_pred_new = fertilizer_meta_model.predict(fertilizer_stacked_predictions_new)

    # Output the final recommendation
    st.write(f"Fertilizer Recommendation for the entered data: {fertilizer_final_pred_new[0]}")

    # Show model performance metrics
    st.subheader("Model Performance Metrics:")
    accuracy = fertilizer_meta_model.score(fertilizer_stacked_predictions_new, fertilizer_final_pred_new)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Assuming the base model has feature importance (e.g., RandomForestClassifier)
    fert_feature_importance = fertilizer_base_models['RandomForestClassifier'].feature_importances_

    # Create a bar chart
    st.subheader("Feature Importance:")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fertilizer_feature_names, fert_feature_importance)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Create a DataFrame from the input data
    st.subheader("Feature Distributions:")
    fertilizer_new_data = pd.DataFrame([[temperature_fertilizer, humidity_fertilizer, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]], columns=fertilizer_feature_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    fertilizer_new_data.plot(kind='bar', ax=ax)
    ax.set_title('Feature Distribution')
    st.pyplot(fig)

    # Plot probabilities
    st.subheader("Prediction Probabilities:")
    fert_predicted_class = fertilizer_final_pred_new[0]
    fert_class_probabilities = fertilizer_stacked_predictions_new[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(fert_class_probabilities)), fert_class_probabilities)
    ax.set_title(f'Prediction Probabilities for {fert_predicted_class}')
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Probability')
    st.pyplot(fig)

    # Add prediction to the DataFrame
    fertilizer_new_data["prediction"] = fertilizer_final_pred_new  

    # Download results as CSV
    st.download_button(label="Download Fertilizer Recommendation Results", data=fertilizer_new_data.to_csv(), file_name="Fertilizer Recommendation Results.csv", mime="text/csv")