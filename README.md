# SMS Spam Detection System

This is a Streamlit web application that uses machine learning to detect spam SMS messages.

## Files
- `sms.py`: The main Streamlit application
- `train_model.py`: Script to train the spam detection model
- `spam.csv`: Dataset containing SMS messages labeled as spam or ham
- `sms_pipeline.pkl`: Trained model pipeline (created after running train_model.py)

## Setup and Running Locally
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Streamlit app:
```bash
streamlit run sms.py
```

## Deployment
The application can be deployed on Streamlit Cloud:
1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy using your forked repository

## Model Details
- Uses TF-IDF vectorization for text processing
- Implements Support Vector Machine (SVM) for classification
- Trained on SMS spam dataset 