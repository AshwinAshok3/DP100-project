# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:04:50 2025

@author: preda
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
from datetime import timedelta
import logging

# Configure logging for the scoring script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global variables for the model and historical data
# These will be loaded once when the endpoint initializes
model = None
historical_data = None
prediction_features = ['lag_1_qty', 'lag_52_qty', 'week_of_year', 'month_of_year', 'year', 'day_of_week']

def init():
    """
    This function is called once when the scoring endpoint is initialized.
    It loads the MLflow model and the historical data needed for feature generation.
    """
    global model, historical_data

    # --- MLflow Model URI Configuration ---
    # In Azure ML deployment, the model path will be provided via an environment variable
    # or directly as part of the model_path argument in deployment configuration.
    # It usually looks like "AZUREML_MODEL_DIR/model_name/version"
    # Or for a registered model, it's typically 'models:/MyForecastModel/Production'
    
    # Attempt to load the model using the path provided by Azure ML or a registered model URI
    model_path = os.getenv("AZUREML_MODEL_DIR", "./model") # Default to './model' for local testing
    
    # If using a registered model name directly in the scoring script, ensure
    # your MLflow tracking URI is set up correctly in Azure ML deployment config.
    # For simplicity, let's assume the model is in a 'model' subdirectory of the deployment
    # (which is common when MLflow models are deployed).
    
    try:
        # Load the MLflow pyfunc model
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # --- Load Historical Data for Lagged Features ---
    # This 'new_dom.csv' is crucial. Ensure it's part of your deployment package.
    # You might place it alongside score.py in your deployment or within the model's artifacts.
    historical_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "new_dom.csv")
    try:
        historical_data = pd.read_csv(historical_data_path, index_col='WeekStarting', parse_dates=True)
        historical_data = historical_data.sort_index()
        logger.info(f"Historical data loaded successfully from: {historical_data_path}")
        logger.info(f"Historical data shape: {historical_data.shape}")
    except FileNotFoundError:
        logger.error(f"Historical data file not found at: {historical_data_path}. This is critical for lagged features.")
        raise
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise

def generate_exogenous_features(target_date: pd.Timestamp, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the exogenous features for a single target date.
    This mimics the feature engineering done during training.
    """
    # Create an empty DataFrame for future exogenous variables
    future_exog_df = pd.DataFrame(index=[target_date])

    # Time-based features
    future_exog_df.loc[target_date, 'week_of_year'] = target_date.isocalendar().week
    future_exog_df.loc[target_date, 'month_of_year'] = target_date.month
    future_exog_df.loc[target_date, 'year'] = target_date.year
    future_exog_df.loc[target_date, 'day_of_week'] = target_date.dayofweek

    # Lagged features: Look up in historical_df
    # Lag 1: Quantity from previous week
    prev_week_date = target_date - timedelta(weeks=1)
    if prev_week_date in historical_df.index:
        future_exog_df.loc[target_date, 'lag_1_qty'] = historical_df.loc[prev_week_date, 'Quantity']
    else:
        logger.warning(f"Actual data for {prev_week_date.date()} not found for lag_1_qty for {target_date.date()}. Filling with NaN.")
        future_exog_df.loc[target_date, 'lag_1_qty'] = np.nan

    # Lag 52: Quantity from 52 weeks ago
    prev_year_date = target_date - timedelta(weeks=52)
    if prev_year_date in historical_df.index:
        future_exog_df.loc[target_date, 'lag_52_qty'] = historical_df.loc[prev_year_date, 'Quantity']
    else:
        logger.warning(f"Actual data for {prev_year_date.date()} not found for lag_52_qty for {target_date.date()}. Filling with NaN.")
        future_exog_df.loc[target_date, 'lag_52_qty'] = np.nan
    
    # Ensure dtypes are consistent with training (important for models like SARIMAX)
    future_exog_df['week_of_year'] = future_exog_df['week_of_year'].astype(int)
    future_exog_df['month_of_year'] = future_exog_df['month_of_year'].astype(int)
    future_exog_df['year'] = future_exog_df['year'].astype(int)
    future_exog_df['day_of_week'] = future_exog_df['day_of_week'].astype(int)

    return future_exog_df[prediction_features] # Ensure correct column order

def run(raw_data):
    """
    This function is called for each batch inference request.
    It takes raw input data (e.g., a list of dates) and returns predictions.

    Args:
        raw_data (str or list of dict): Input data received by the endpoint.
                                       For batch, this is often a JSON string
                                       representing a list of inputs.
                                       Example: '{"input_dates": ["1992-10-08"]}'
                                       Or a list of paths to input files.
                                       For simplicity, we'll assume a JSON string
                                       containing a list of dates.

    Returns:
        json: A JSON string containing the predictions.
    """
    logger.info("Batch inference request received.")
    
    try:
        # Parse the input data. For a batch endpoint, this could be more complex,
        # e.g., reading from a path, or directly receiving a Pandas DataFrame.
        # Assuming the input is a JSON string with a key 'input_dates'
        data = json.loads(raw_data)
        input_dates_str = data.get("input_dates", [])
        
        if not input_dates_str:
            return json.dumps({"error": "No 'input_dates' provided in the input."})

        all_predictions = []
        for date_str in input_dates_str:
            try:
                target_date = pd.to_datetime(date_str)
                
                # Generate exogenous features for the target date
                # We need the full historical_data available globally for lags
                exog_for_prediction = generate_exogenous_features(target_date, historical_data)

                # Check for NaNs in exog, which indicate missing historical data for lags
                if exog_for_prediction.isnull().any().any():
                    logger.warning(f"NaNs found in exogenous features for {date_str}. Cannot predict accurately.")
                    # You might choose to skip this prediction or impute NaNs
                    predicted_quantity = None # Indicate failure for this date
                else:
                    # The 'predict' method of SARIMAX requires the index based on the original series length
                    # For a single future step, it's typically one step beyond the last known historical data point.
                    # This relies on the 'model' object implicitly knowing its training data length.
                    
                    # Instead of precise 'start'/'end' indices for each prediction (which is complicated
                    # for arbitrary future dates without context of original series length),
                    # for a single point prediction with exog, SARIMAX.predict() can sometimes infer if it's one step out.
                    # A more robust way for future prediction is `model.forecast(steps=1, exog=exog_for_prediction)`
                    
                    # The `loaded_model` from `mlflow.pyfunc.load_model` will have a `predict` method.
                    # It's usually best to format input as a DataFrame that matches the training X.
                    
                    # Using `forecast` is often simpler for out-of-sample one-step predictions
                    # It assumes the `loaded_model` handles the current state correctly.
                    
                    # For a pyfunc model, it's simpler to pass the dataframe directly.
                    # The pyfunc `predict` will handle the SARIMAX `forecast` or `predict` internally.
                    prediction_result = model.predict(exog_for_prediction) # Pass the DataFrame of exogenous vars
                    predicted_quantity = prediction_result.iloc[0] # Get the first (and only) prediction

                all_predictions.append({
                    "date": date_str,
                    "predicted_quantity": predicted_quantity
                })
                logger.info(f"Prediction for {date_str}: {predicted_quantity}")

            except Exception as date_e:
                logger.error(f"Error processing date {date_str}: {date_e}")
                all_predictions.append({"date": date_str, "predicted_quantity": None, "error": str(date_e)})

        return json.dumps(all_predictions)

    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        return json.dumps({"error": str(e)})