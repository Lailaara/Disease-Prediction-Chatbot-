import streamlit as st
import openai
import pandas as pd
import joblib

# Set your OpenAI API key
openai.api_key = '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'

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Define the report generation function using GPT
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

    # Call OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # GPT model
        prompt=prompt,
        max_tokens=300  # Adjust token limit as needed
    )

    # Extract the text part of the response
    report = response.choices[0].text.strip()
    return report

# Define prognosis mapping
prognosis_mapping = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    4: 'Drug Reaction', 5: 'Peptic ulcer disease', 6: 'AIDS', 7: 'Diabetes',
    # Add other mappings...
}

# Streamlit UI
st.title("Disease Prediction App")

# Symptom selection
correct_feature_names = ['itching', 'skin_rash', 'vomiting', 'headache']  # Example features
selected_symptoms = st.multiselect('Select symptoms you are experiencing:', correct_feature_names)

# Prediction
if st.button('Predict Disease'):
    input_features = [1 if symptom in selected_symptoms else 0 for symptom in correct_feature_names]
    input_df = pd.DataFrame([input_features], columns=correct_feature_names)

    # Predict the disease
    prediction = model.predict(input_df)
    predicted_prognosis = prognosis_mapping.get(prediction[0], "Unknown Prognosis")

    # Display the predicted disease
    st.write(f'Predicted Disease: **{predicted_prognosis}**')

    # Generate and display the detailed disease report using GPT
    if predicted_prognosis != "Unknown Prognosis":
        report = generate_disease_report(predicted_prognosis)
        st.write(f'Detailed Report:\n{report}')

        # Create a downloadable link for the report
        st.download_button(
            label="Download Report as Text",
            data=report,
            file_name=f"{predicted_prognosis}_report.txt",
            mime="text/plain"
        )
    else:
        st.write("The disease prediction is unknown, please consult a healthcare provider.")
