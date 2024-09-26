import streamlit as st
import openai
import pandas as pd
import joblib

# Set your OpenAI API key
openai.api_key = 'asst_hK1PKll1Y2NqovYCWZmz2vgR'

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Get feature names used during model training
try:
    feature_names = model.feature_names_in_  # Try to get feature names from the model
except AttributeError:
    # Manually define or handle missing feature names (based on how the model was trained)
    feature_names = [
        'itching', 'skin_rash', 'abdominal_pain', 'nausea', 'vomiting', 'headache', 'fever',  # Add all training features here
        'joint_pain', 'cough', 'fatigue', 'diarrhoea', 'sore_throat', 'shortness_of_breath'
    ]

# Define the report generation function using ChatCompletion API
def generate_disease_report(disease_name):
    prompt = f"""
    Provide a detailed report for the disease {disease_name}. 
    Include:
    1. A brief explanation of the disease.
    2. Common symptoms.
    3. Recommended treatments.
    4. Suggested lifestyle changes.
    5. When to consult a doctor.
    """
    
    # Use ChatCompletion instead of Completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 or GPT-4 as needed
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Get the generated response
    report = response['choices'][0]['message']['content'].strip()
    return report

# Define prognosis mapping (replace with actual mappings)
prognosis_mapping = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    # Add other mappings...
}

# Streamlit UI
st.title("Symptom Checker Chatbot")

# Symptom selection
selected_symptoms = st.multiselect('Select symptoms you are experiencing:', feature_names)

# Prediction
if st.button('Predict Disease'):
    # Ensure the input_df has the same columns (features) as used during training
    input_features = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in feature_names}
    
    # Create DataFrame from input features
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # Debug: Check if the input DataFrame is correct
    st.write("Input DataFrame for Prediction:", input_df)

    # Predict the disease
    try:
        prediction = model.predict(input_df)
        predicted_prognosis = prognosis_mapping.get(prediction[0], "Unknown Prognosis")

        # Debug: Check the prediction result
        st.write("Raw Prediction Result:", prediction)
        st.write(f'Predicted Disease: **{predicted_prognosis}**')

        # Generate and display the detailed disease report using GPT
        if predicted_prognosis != "Unknown Prognosis":
            report = generate_disease_report(predicted_prognosis)
            st.write(f'Detailed Report:\n{report}')
        else:
            st.write("The disease prediction is unknown, please consult a healthcare provider.")

    except Exception as e:
        # Display any error during prediction
        st.write(f"Error during prediction: {e}")


