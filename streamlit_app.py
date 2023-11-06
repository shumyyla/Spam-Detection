import streamlit as st
import pickle

with open('spam_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Spam Detection App")

user_input = st.text_area("Enter a text for  detection:")

predict_button = st.button("Predict")

if user_input and predict_button:
    user_input_vectorized = model.named_steps['tfidf'].transform([user_input])
    prediction = model.named_steps['classifier'].predict(user_input_vectorized)
    label = 'Spam' if prediction == 1 else 'Ham'
    st.write(f"The text is classified as : {label}")
