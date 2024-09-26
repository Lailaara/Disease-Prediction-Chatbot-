import openai
import pickle
import numpy as np

# Load your OpenAI API key (replace with your actual key)
openai.api_key = "sk-svcacct-hI0CpoXDOAN0BXvT8DmhBrd1Lwphioo-FMmSG4HuY__phm0E0gm6PHXngIcT3BlbkFJKTbhHsCFf-FjPR1C_19tO9lmDv9mWWjDb8CJcNile-ne2PEFB1ae8-PUdIQA"

# Function to interact with GPT-4 using the new interface
def ask_gpt(question):
    try:
        # In the new interface, you may need to use a different method
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
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"

# Load your disease prediction model (assuming it has already been trained)
with open('disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example function to predict disease and answer user queries
def predict_and_answer(features, user_query=None):
    try:
        # Reshape features for the model (adjust as necessary)
        features = np.array(features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)[0]
        
        # Format prediction result
        result = f"The model predicts: {prediction}"
        
        # If the user asks a question, query GPT-4 for an explanation or further details
        if user_query:
            gpt_response = ask_gpt(f"Based on the prediction '{prediction}', {user_query}")
            return result, gpt_response
        
        return result
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Example feature input (replace with actual data input mechanism)
    example_features = [5.1, 3.5, 1.4, 0.2]  # Replace with actual input feature vector

    # Simulate a user query
    user_query = "Can you explain what this prediction means?"

    # Get prediction and LLM response
    prediction, explanation = predict_and_answer(example_features, user_query)
    
    # Output the results
    print("Prediction:", prediction)
    print("Explanation from LLM:", explanation)
