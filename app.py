import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# --- Background Image ---
BACKGROUND_IMAGE_URL = "https://media.istockphoto.com/id/1069118084/photo/golden-background-ingots-or-nuggets-of-pure-gold-gold-leaf-tea-resin-puer.jpg?s=612x612&w=0&k=20&c=gOTRMnAoGup-np-35jz-5Cg-fPVJD3Q8l_7Us3FO3mk="
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({BACKGROUND_IMAGE_URL});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stDecoration{{
        opacity: 0}}
    .stAppToolbar{{
        background: rgba(0, 0, 0, 0.0) !important;
        }}
    .st-emotion-cache-1j22a0y{{
        opacity:1}}
    /* Make the main content area fully transparent */
    .stApp > div {{
        background-color: rgba(30, 30, 30, 0.0); /* Fully transparent background for main content */
        padding: 1rem;
        border-radius: 10px;
        color: white; /* Ensure text is readable against the background */
    }}
    /* Target the Streamlit header bar itself for transparency */
    /* This class (or similar like .st-emotion-cache-uf99v8 or header) often controls the top bar */
    .st-emotion-cache-z5rdx8 {{ /* This class targets the top header bar itself */
        background-color: rgba(30, 30, 30, 0.0) !important; /* Make header transparent */
    }}
    .st-emotion-cache-uf99v8 {{ /* Another common class for the header container */
        background-color: rgba(30, 30, 30, 0.0) !important;
    }}
    /* Adjust text color for better readability on a darker background */
    .stMarkdown, .stLabel, .stSlider > div > div > div > div {{
        color: white;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: gold; /* Make titles stand out */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.7) !important; /* Slightly transparent white for alerts */
        border-radius: 10px;
        color: black; /* Ensure alert text is readable */
    }}
    .stButton > button {{
        background-color: #FFD700; /* Gold color for button */
        color: black;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }}
    .stButton > button:hover {{
        background-color: #DAA520; /* Darker gold on hover */
        color: black;
        border-color: #DAA520;
    }}
    /* Style for number inputs and selectboxes */
    .stNumberInput > div > div > input, .stSelectbox > div > div > div > div > div {{
        background-color: rgba(255, 255, 255, 0.2); /* Transparent white input background */
        color: white;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the Trained Model ---
model_path = 'trained_model.sav'

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found.")
    st.warning("Please run 'train_model.py' first to create the model file.")
    st.stop()

try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully! Ready for predictions.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Please verify the model file's integrity and ensure it was saved correctly.")
    st.stop()

# --- Prediction Function ---
def predict_price(input_features):
    # The loaded_model is a Pipeline (scaler + regressor).
    # It expects a DataFrame-like input for consistency.
    feature_names = ['SPX', 'GLD', 'USO', 'SLV']
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # The pipeline's scaler will automatically scale these features before prediction
    prediction = loaded_model.predict(input_df)[0]
    return prediction

# --- Streamlit App Main Function ---
def main():
    st.set_page_config(page_title="Price Predictor (USD/EUR)", layout="centered")

    st.title("💰 Price Prediction (USD/EUR) Web App")
    st.markdown("Enter the market indices and commodity prices to predict a target price.")

    st.header("Input Features")

    # Input fields for features
    spx = st.slider('SPX (S&P 500 Index)', min_value=1000.0, max_value=6000.0, value=4000.0, step=1.0)
    gld = st.slider('GLD (Gold ETF Price)', min_value=100.0, max_value=300.0, value=170.0, step=0.1)
    uso = st.slider('USO (Oil ETF Price)', min_value=10.0, max_value=100.0, value=50.0, step=0.1)
    slv = st.slider('SLV (Silver ETF Price)', min_value=10.0, max_value=40.0, value=22.0, step=0.1)

    # Button to trigger prediction
    if st.button('Predict Price'):
        # Prepare input data as a list
        input_data = [spx, gld, uso, slv]

        try:
            predicted_value = predict_price(input_data)
            st.success(f"Estimated Price: USD/EUR {predicted_value:,.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please check the input values and model compatibility.")

    st.markdown("---")
    st.caption("Powered by Machine Learning")

# --- Run the Streamlit App ---
if __name__ == '__main__':
    main()
