import openai
import pandas as pd
import joblib

# Set your OpenAI API key
openai.api_key = 'asst_hK1PKll1Y2NqovYCWZmz2vgR'  # Replace with your actual API key

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

# Define the report generation function using GPT-3 (text-davinci-003 model)
def generate_disease_report(disease_name):
    # Define the prompt for generating a disease report
    prompt = f"""
    Provide a detailed report for the disease {disease_name}. 
    Include:
    1. A brief explanation of the disease.
    2. Common symptoms.
    3. Recommended treatments.
    4. Suggested lifestyle changes.
    5. When to consult a doctor.
    """
    
    # Use the newer API method with text-davinci-003 model
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    
    # Extract and return the content of the response
    return response.choices[0].text.strip()

# Define prognosis mapping (replace with actual mappings)
prognosis_mapping = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    # Add other mappings...
}

# Function to gather user input (symptoms) from the command line
def get_user_symptoms():
    print("Please select symptoms from the following list (separate by commas):")
    print(", ".join(feature_names))
    
    user_input = input("Enter your symptoms: ").lower().split(", ")
    selected_symptoms = [symptom.strip() for symptom in user_input if symptom in feature_names]
    
    if not selected_symptoms:
        print("No valid symptoms entered. Please try again.")
        return get_user_symptoms()
    
    return selected_symptoms

# Main function for disease prediction
def predict_disease():
    # Get symptoms from user input
    selected_symptoms = get_user_symptoms()
    
    # Ensure the input_df has the same columns (features) as used during training
    input_features = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in feature_names}
    
    # Create DataFrame from input features
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    # Display the input data for prediction (for debugging purposes)
    print("Input DataFrame for Prediction:")
    print(input_df)
    
    try:
        # Predict the disease
        prediction = model.predict(input_df)
        predicted_prognosis = prognosis_mapping.get(prediction[0], "Unknown Prognosis")
        
        # Display the predicted disease
        print(f'Predicted Disease: {predicted_prognosis}')
        
        # Generate and display the detailed disease report using GPT
        if predicted_prognosis != "Unknown Prognosis":
            report = generate_disease_report(predicted_prognosis)
            print(f'\nDetailed Report:\n{report}')
        else:
            print("The disease prediction is unknown, please consult a healthcare provider.")

    except Exception as e:
        # Display any error during prediction
        print(f"Error during prediction: {e}")

# Run the disease prediction
if __name__ == "__main__":
    predict_disease()
