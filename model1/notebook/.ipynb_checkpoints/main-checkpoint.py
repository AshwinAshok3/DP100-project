# Import necessary libraries
import pandas as pd
import numpy as np
import os
import argparse
import logging

# Azure ML SDK imports
from azureml.core import Run, Dataset, Model
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.forecasting_parameters import ForecastingParameters

# MLflow imports
import mlflow
import mlflow.azureml

# Configure logging for better visibility in Azure ML logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(input_dataset):
    """
    Loads data from the given input dataset and performs basic preparation.

    Args:
        input_dataset (azureml.core.Dataset): The dataset passed to the script.

    Returns:
        pandas.DataFrame: The prepared DataFrame.
    """
    logger.info("Loading data from input dataset...")
    # Convert the Azure ML Dataset to a Pandas DataFrame.
    # This efficiently streams the data from Azure storage to the compute cluster.
    df = input_dataset.to_pandas_dataframe()

    logger.info("Converting 'WeekStarting' to datetime and sorting data...")
    # Convert 'WeekStarting' column to datetime objects.
    df['WeekStarting'] = pd.to_datetime(df['WeekStarting'])
    # Sort the data by store, brand, and then week to ensure chronological order for each series.
    df = df.sort_values(by=['Store', 'Brand', 'WeekStarting'])

    logger.info(f"Data loaded and prepared. Shape: {df.shape}")
    logger.info("Data head:\n" + str(df.head()))
    return df


def train_forecasting_model_with_automl(run, training_data_df, target_column, time_column, time_series_ids,
                                        forecast_horizon, experiment_timeout_minutes=20):
    """
    Configures and runs an Automated ML experiment for time series forecasting.

    Args:
        run (azureml.core.Run): The current Azure ML run context.
        training_data_df (pandas.DataFrame): The DataFrame containing training data.
        target_column (str): Name of the column to forecast.
        time_column (str): Name of the time column.
        time_series_ids (list): List of columns identifying unique time series (e.g., ['Store', 'Brand']).
        forecast_horizon (int): Number of periods to forecast into the future.
        experiment_timeout_minutes (int): Maximum time in minutes for the AutoML experiment to run.

    Returns:
        tuple: (best_automl_model, best_run_metrics, best_run)
    """
    logger.info(f"Starting AutoML forecasting experiment for target '{target_column}'...")

    # Define forecasting parameters for AutoML
    forecasting_parameters = ForecastingParameters(
        time_column_name=time_column,
        forecast_horizon=forecast_horizon,
        # IMPORTANT: This tells AutoML to treat 'Store' and 'Brand' as distinct time series.
        time_series_id_column_names=time_series_ids,
        # Optional: You can specify frequency if auto-detection is not reliable (e.g., 'W' for weekly)
        # freq='W',
        # Optional: Add target lags or rolling window features (AutoML can auto-generate these)
        # target_lags='auto',
        # target_rolling_window_size='auto'
    )

    # Configure the AutoML experiment
    automl_config = AutoMLConfig(
        task='forecasting',
        # AutoML takes a TabularDataset. We create it on-the-fly from the Pandas DataFrame.
        # For larger datasets, it's better to create a registered TabularDataset beforehand.
        training_data=Dataset.Tabular.register_pandas_dataframe(training_data_df,
                                                                run.experiment.workspace.get_default_datastore(),
                                                                'automl_temp_train_data'),
        target_column_name=target_column,
        primary_metric='normalized_root_mean_squared_error',  # Common metric for forecasting
        forecasting_parameters=forecasting_parameters,
        # AutoML will run on the cluster this script is submitted to
        compute_target=run.experiment.workspace.compute_targets[run.compute_name],
        experiment_timeout_minutes=experiment_timeout_minutes,
        # Maximum concurrent AutoML iterations (trials) on the cluster
        max_concurrent_iterations=os.environ.get('AML_VC_PROCESS_COUNT', 1),
        # Use environment variable for cluster parallelism
        enable_early_stopping=True,
        # You can block specific models if they are not suitable or too slow
        # blocked_models=['ARIMA', 'Prophet'], # Example: to exclude certain models
        # enable_dnn_forecasting=True, # Enable deep learning models for forecasting
        featurization='auto'  # Let AutoML handle feature engineering automatically
    )

    # Submit the AutoML run. This will create child runs within the main experiment.
    automl_run = run.experiment.submit(automl_config, show_output=True)
    logger.info(f"AutoML run submitted. Details at: {automl_run.get_portal_url()}")

    # Wait for the AutoML run to complete
    automl_run.wait_for_completion(show_output=True)

    # Retrieve the best model from the AutoML run
    best_automl_model, best_run = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()

    logger.info(f"Best AutoML model found: {best_automl_model}")
    logger.info(f"Best AutoML run ID: {best_run.id}")
    logger.info(f"Best AutoML run metrics: {best_run_metrics}")

    return best_automl_model, best_run_metrics, best_run


def log_metrics_with_mlflow(run, metrics, prefix="automl_"):
    """
    Logs metrics to MLflow from the current Azure ML run context.

    Args:
        run (azureml.core.Run): The current Azure ML run context.
        metrics (dict): Dictionary of metrics to log.
        prefix (str): Prefix to add to metric names (e.g., 'automl_rmse').
    """
    logger.info("Logging metrics to MLflow...")
    with mlflow.start_run(run_id=run.id):  # Continue the existing Azure ML run in MLflow
        for key, value in metrics.items():
            # Only log numerical values
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}{key}", value)
                logger.info(f"  Logged MLflow metric: {prefix}{key} = {value}")
            else:
                logger.info(f"  Skipped non-numerical metric: {key}")


def register_automl_model(run, best_model, model_name="oj_sales_automl_forecaster", tags=None):
    """
    Registers the best AutoML model in the Azure ML Model Registry.

    Args:
        run (azureml.core.Run): The current Azure ML run context.
        best_model: The best model object returned by AutoML.
        model_name (str): Name to register the model under.
        tags (dict): Optional dictionary of tags for the registered model.

    Returns:
        azureml.core.Model: The registered model object.
    """
    logger.info(f"Registering best AutoML model as '{model_name}'...")
    # The best_model from AutoML is already a Model object or can be converted.
    # AutoML often registers the model automatically. We can get it directly from best_run.

    # Retrieve the model from the best AutoML run and register it
    registered_model = best_model.register(
        workspace=run.experiment.workspace,
        model_name=model_name,
        tags=tags if tags else {},
        description="AutoML time series forecasting model for OJ Sales"
    )
    logger.info(f"Model registered. Name: {registered_model.name}, Version: {registered_model.version}")
    return registered_model


def main():
    """
    Main function to orchestrate the forecasting process.
    """
    logger.info("--- my_training_script.py: Main function started ---")

    # Get the current Azure ML Run context. This is how the script interacts with Azure ML.
    run = Run.get_context()

    # Get the input dataset. 'aml_data' is the name we'll use when defining the input in the notebook.
    # Ensure this name matches the input mapping in your ScriptRunConfig.
    if 'aml_data' not in run.input_datasets:
        logger.error("Input dataset 'aml_data' not found. Please ensure it's passed correctly.")
        raise ValueError("Missing input dataset 'aml_data'.")

    input_dataset = run.input_datasets['aml_data']

    # --- Step 1: Load and Prepare Data ---
    training_data_df = load_and_prepare_data(input_dataset)

    # --- Step 2: Run Automated ML for Forecasting ---
    # Define forecasting parameters
    target_column = 'Quantity'
    time_column = 'WeekStarting'
    time_series_ids = ['Store', 'Brand']  # Crucial for multi-series forecasting
    forecast_horizon = 10  # Predict 10 weeks into the future
    experiment_timeout_minutes = 30  # Set a reasonable timeout for the AutoML experiment

    best_automl_model, best_run_metrics, best_run = train_forecasting_model_with_automl(
        run, training_data_df, target_column, time_column, time_series_ids, forecast_horizon, experiment_timeout_minutes
    )

    # --- Step 3: Log Metrics with MLflow ---
    # MLflow automatically logs many metrics from AutoML. We can log additional custom ones if needed.
    # Here, we'll log the best run's metrics directly via MLflow.
    # Note: Many of these are already logged by AutoML itself to the parent run and child runs.
    log_metrics_with_mlflow(run, best_run_metrics)

    # --- Step 4: Register the Best AutoML Model ---
    # AutoML also registers the best model by default. This step is to explicitly show how to do it
    # or to register it with custom tags/name if needed.
    model_tags = {'task': 'forecasting', 'dataset': 'OjSalesSimulated', 'best_model_run_id': best_run.id}
    registered_model = register_automl_model(run, best_automl_model, tags=model_tags)

    logger.info("--- my_training_script.py: Main function finished ---")


if __name__ == '__main__':
    # This block executes when the script is run directly (e.g., by Azure ML).
    # It calls the main orchestration function.
    main()