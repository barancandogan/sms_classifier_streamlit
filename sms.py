import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱"
)

# Title and description
st.title("SMS Spam Detection System")
st.write("This application helps you detect whether a text message is spam or not.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('sms_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the model
model = load_model()

# Text input section
text_input = st.text_area("Enter the message to analyze:", height=100)

if st.button("Analyze Text"):
    if text_input.strip() != "":
        if model is not None:
            try:
                # Make prediction
                prediction = model.predict([text_input])[0]
                probability = model.predict_proba([text_input])[0]
                
                # Get the probability of the predicted class
                spam_prob = probability[1] if prediction == 1 else probability[0]
                
                # Display results
                st.write("### Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    prediction_label = "SPAM" if prediction == 1 else "NOT SPAM"
                    st.metric("Prediction", prediction_label)
                
                with col2:
                    st.metric("Confidence Score", f"{spam_prob:.2%}")
                
                # Add additional information
                if prediction == 1:
                    st.warning("⚠️ This message has been classified as SPAM!")
                else:
                    st.success("✅ This message appears to be legitimate.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.error("Model could not be loaded. Please make sure the model is trained and 'sms_pipeline.pkl' exists.")
    else:
        st.warning("Please enter some text to analyze.")

# Add information about the application
with st.expander("About this app"):
    st.write("""
    This SMS Spam Detection system uses a Support Vector Machine (SVM) model to classify text messages as spam or not spam.
    
    To use:
    1. Enter your text message in the text area
    2. Click 'Analyze Text'
    3. View the prediction and confidence score
    
    The model uses TF-IDF vectorization for text processing and SVM for classification.
    """)
