import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.preprocessor import create_preprocessing_pipeline
from src.load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import set_config


def load_model():
    lgbm_model = joblib.load('models/lgbm_model.joblib')
    gbr_model = joblib.load('models/gbr_model.joblib')

    return lgbm_model, gbr_model

def load_and_fit_preprocessor(file_path):
    housing_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(housing_df)

    return preprocessor 

def user_input_features():
    inputs = {
        'SqFtTotLiving': st.sidebar.number_input('Total Living Area (SqFt)', 0, 7000, 0),
        'Latitude': st.sidebar.number_input('Latitude', -90.0, 90.0, 47.0),
        'Longitude': st.sidebar.number_input('Longitude', -180.0, 180.0, -122.0),
        'SqFt2ndFloor': st.sidebar.number_input('Second Floor Area (SqFt)', 0, 5000, 0),
        'YrBuilt': st.sidebar.number_input('Year Built', 0, 2019, 0),
        'SqFtFinBasement': st.sidebar.number_input('Finished Basement Area (SqFt)', 0, 5000, 0),
        'SqFt1stFloor': st.sidebar.number_input('First Floor Area (SqFt)', 0, 5000, 0),

    }
    input_df = pd.DataFrame(inputs, index=[0])
    st.write("User Input Features:")
    st.write(input_df)
    return input_df

def make_prediction(model, preprocessor, input_df):
    input_df_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_df_transformed)
    st.write("Model Prediction:")
    st.write(prediction)
    return prediction[0]


st.image('images/seattle.jpeg', use_column_width=True)

st.title('House Price Prediction App')

# Load models and preprocessor
lgbm_model, gbr_model = load_model()
preprocessor = load_and_fit_preprocessor('data/transformed/new_housing_df.csv')

# Get user input
input_df = user_input_features()

# Choose model
model_choice = st.sidebar.selectbox('Choose Model', ('LGBM', 'GBR'))
if model_choice == 'LGBM':
    model = lgbm_model
else:
    model = gbr_model
st.write(f'### Selected model: {model_choice}')

# Make prediction
if st.button('Predict'):
    prediction = make_prediction(model, preprocessor, input_df)
    st.write(f'The predicted house price is: ${prediction:,.2f}')