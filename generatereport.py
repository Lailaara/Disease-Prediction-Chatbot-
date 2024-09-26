import openai
import streamlit as st
import pickle
import numpy as np

# Load your OpenAI API key
openai.api_key = "sk-svcacct-hI0CpoXDOAN0BXvT8DmhBrd1Lwphioo-FMmSG4HuY__phm0E0gm6PHXngIcT3BlbkFJKTbhHsCFf-FjPR1C_19tO9lmDv9mWWjDb8CJcNile-ne2PEFB1ae8-PUdIQA"

# Function to interact with GPT-4
def ask_gpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Load the disease prediction model
with open('disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app interface
st.title("Disease Prediction with LLM Integration")

# Input form for features
features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(5)]  # Adjust based on feature count

# Predict button
if st.button("Predict"):
    prediction = model.predict([features])
    st.write(f"Prediction: {prediction}")

    # Allow user to ask questions
    user_query = st.text_input("Ask a question related to the prediction")
    if user_query:
        response = ask_gpt(user_query)
        st.write(f"LLM Response: {response}")
