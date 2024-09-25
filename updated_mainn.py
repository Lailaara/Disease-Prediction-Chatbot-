
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Extract correct feature names from the model (if available)
try:
    correct_feature_names = model.feature_names_in_
except AttributeError:
    correct_feature_names = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
        'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
        'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
        'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
        'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
        'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
        'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
        'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
        'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
        'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
        'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
        'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
        'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
        'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
        'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
        'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
        'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
        'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
        'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
        'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
        'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]

# Full mapping of numerical codes to prognosis labels
prognosis_mapping = {0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis', 4: 'Drug Reaction', 5: 'Peptic ulcer diseae', 6: 'AIDS', 7: 'Diabetes ', 8: 'Gastroenteritis', 9: 'Bronchial Asthma', 10: 'Hypertension ', 11: 'Migraine', 12: 'Cervical spondylosis', 13: 'Paralysis (brain hemorrhage)', 14: 'Jaundice', 15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid', 19: 'hepatitis A', 20: 'Hepatitis B', 21: 'Hepatitis C', 22: 'Hepatitis D', 23: 'Hepatitis E', 24: 'Alcoholic hepatitis', 25: 'Tuberculosis', 26: 'Common Cold', 27: 'Pneumonia', 28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack', 30: 'Varicose veins', 31: 'Hypothyroidism', 32: 'Hyperthyroidism', 33: 'Hypoglycemia', 34: 'Osteoarthristis', 35: 'Arthritis', 36: '(vertigo) Paroymsal  Positional Vertigo', 37: 'Acne', 38: 'Urinary tract infection', 39: 'Psoriasis', 40: 'Impetigo'}

# Streamlit UI
st.title("Disease Prediction App")

# Symptom selection
selected_symptoms = st.multiselect('Select symptoms you are experiencing:', correct_feature_names)

# Prediction
if st.button('Predict Disease'):
    input_features = [1 if symptom in selected_symptoms else 0 for symptom in correct_feature_names]
    input_df = pd.DataFrame([input_features], columns=correct_feature_names)

    # Predict the disease
    prediction = model.predict(input_df)
    predicted_prognosis = prognosis_mapping.get(prediction[0], "Unknown Prognosis")

    # Display the result
    st.write(f'Predicted Disease: **{predicted_prognosis}**')
