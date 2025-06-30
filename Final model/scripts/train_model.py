# importing libraries required
import logging
import importlib
import sys
import os 
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import mlflow
warnings.filterwarnings("ignore") # Suppress warnings for cleaner output

# --- Basic Logging Configuration (Assumes this is run once at application startup) ---
# This sets up how logs will be formatted and where they go.
# It's placed here to ensure logging is configured when the module is loaded.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Logger for this module



# Data preparation and preprocessing with data cleaning function
def data_prep(df):
    # Start a nested MLflow run for data preparation.
    # This ensures that if there's an outer run already active,
    # data_prep's activities are logged as a child run.
    # If no outer run is active, it will start a new top-level run for this function.
    with mlflow.start_run(run_name="Data Preprocessing", nested=True):
        logger.info("Starting data preparation and preprocessing.")

        new_dict = {}
        new_dict["WeekStarting"] = df["WeekStarting"].copy()
        new_dict["dom_qty"] = df["Quantity"].copy()
        dom_df = pd.DataFrame(new_dict)
        
        dom_df["WeekStarting"] = pd.to_datetime(dom_df["WeekStarting"])
        dom_df.set_index("WeekStarting",inplace=True)
        dom_df = dom_df.sort_index()
        
        dom_df['lag_1_dom_qty'] = dom_df['dom_qty'].shift(1)
        dom_df['lag_52_dom_qty'] = dom_df['dom_qty'].shift(52)
        
        # Create time-based features from the index
        # These features help the model capture trends and seasonal components more explicitly.
        dom_df['week_of_year'] = dom_df.index.isocalendar().week.astype(int) # Week of the year (1-52/53)
        dom_df['month_of_year'] = dom_df.index.month # Month of the year (1-12)
        dom_df['year'] = dom_df.index.year # Year (for trend)
        dom_df['day_of_week'] = dom_df.index.dayofweek # Day of the week (0=Monday, 6=Sunday)
        
        # Log parameters or artifacts
        mlflow.log_param("initial_rows", len(df))
        mlflow.log_param("processed_rows", len(dom_df))
        
        logger.info("Data preparation and preprocessing completed successfully.")
        
        # Optionally, you can also log the processed data as an artifact if it's not too large
        # from mlflow.models import infer_signature
        # mlflow.log_artifact(local_path_to_save_df, "processed_data.parquet") 
        # For this, you'd need to save dom_df to a file first.
        
        return dom_df


# Function for partitioning the train and test set
def data_split(dom_df):
    # Set MLflow tracking URI and experiment. This will ensure MLflow is initialized
    # and either connects to an existing experiment or creates a new one.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Data Preparation Experiment") # Or a more specific experiment name if desired

    # Start an MLflow run. Using 'nested=True' is crucial here, as this function
    # will likely be called within a larger MLflow run (e.g., in a main training script).
    # If no run is active, it will start a new top-level run.
    with mlflow.start_run(run_name="Data Splitting", nested=True):
        logger.info("Starting data splitting process.")

        # Drop rows with NaNs introduced by lagging before splitting
        df_clean = dom_df.dropna()
        
        # Use about 80% for training and 20% for testing
        # This ensures that the test set still has future data not seen by the model.
        train_size = int(len(df_clean) * 0.8)
        train_data_df = df_clean.iloc[0:train_size]
        test_data_df = df_clean.iloc[train_size:]
        
        # Separate endogenous variable (y) and exogenous variables (X) for SARIMAX
        y_train = train_data_df['dom_qty']
        X_train = train_data_df[['lag_1_dom_qty', 'lag_52_dom_qty', 'week_of_year', 'month_of_year', 'year', 'day_of_week']]
        
        y_test = test_data_df['dom_qty']
        X_test = test_data_df[['lag_1_dom_qty', 'lag_52_dom_qty', 'week_of_year', 'month_of_year', 'year', 'day_of_week']]
        
        # Log data split parameters
        mlflow.log_param("train_size_ratio", 0.8)
        mlflow.log_param("train_data_points", len(train_data_df))
        mlflow.log_param("test_data_points", len(test_data_df))

        # Log info messages with the logger
        logger.info(f"Training data points (after dropping NaNs): {len(train_data_df)}")
        logger.info(f"Testing data points (after dropping NaNs): {len(test_data_df)}\n")

        # The print statements are retained as per your request to "dont change the code"
        print(f"Training data points (after dropping NaNs): {len(train_data_df)}")
        print(f"Testing data points (after dropping NaNs): {len(test_data_df)}\n")
        
        logger.info("Data splitting completed successfully.")
        return X_train, y_train, X_test, y_test


# function for hyperparameter tuning
def hp_tuning_model(X_train, y_train):
    # Set MLflow tracking URI and experiment. This ensures MLflow is initialized
    # and either connects to an existing experiment or creates a new one.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Model Hyperparameter Tuning Experiment") # New experiment name for this stage

    # Start an MLflow run. Using 'nested=True' is crucial here, as this function
    # will likely be called within a larger MLflow run (e.g., in a main training script).
    # If no run is active, it will start a new top-level run.
    with mlflow.start_run(run_name="SARIMAX Hyperparameter Tuning", nested=True):
        logger.info("Starting SARIMAX hyperparameter tuning.")
        
        # Define ranges for non-seasonal parameters (p, d, q)
        p_range = d_range = q_range = range(0, 3) # Test 0, 1, 2
        non_seasonal_pdq = list(product(p_range, d_range, q_range))
        
        # Define ranges for seasonal parameters (P, D, Q)
        P_range = D_range = Q_range = [0, 1] # Typically 0 or 1 for seasonal components
        seasonal_period = 52 # Given weekly data, yearly seasonality is 52 weeks
        seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(product(P_range, D_range, Q_range))]
        
        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        best_model_fit = None
        
        # Original print statements retained
        print("--- Starting SARIMAX Hyperparameter Tuning (This may take a while) ---")
        print("Evaluating models based on AIC. Lower AIC is better.")
        
        logger.info("Iterating through SARIMAX orders for hyperparameter tuning.")

        for order in non_seasonal_pdq:
            for s_order in seasonal_pdq:
                try:
                    model = SARIMAX(y_train,
                                    exog=X_train, # Include engineered features as exogenous variables
                                    order=order,
                                    seasonal_order=s_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    model_fit = model.fit(disp=False, maxiter=100) # disp=False suppresses fitting output
            
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = order
                        best_seasonal_order = s_order
                        best_model_fit = model_fit
                    # print(f"SARIMAX{order}{s_order} - AIC: {model_fit.aic:.2f}") # Uncomment to see progress
                except Exception as e:
                    logger.warning(f"SARIMAX{order}{s_order} - Error during fit: {e}") # Log errors
                    # print(f"SARIMAX{order}{s_order} - Error: {e}") # Uncomment to see errors for specific orders
                    continue
        
        # Log the best hyperparameters found
        mlflow.log_param("best_aic", best_aic)
        mlflow.log_param("best_non_seasonal_order", best_order)
        mlflow.log_param("best_seasonal_order", best_seasonal_order)
        mlflow.log_param("seasonal_period", seasonal_period)

        logger.info(f"Hyperparameter tuning completed. Best AIC: {best_aic:.2f}")
        logger.info(f"Best Non-Seasonal Order: {best_order}")
        logger.info(f"Best Seasonal Order: {best_seasonal_order}")

        return best_aic, best_order, best_seasonal_order, best_model_fit, seasonal_period



# training the model with the best parameters
def model_training(best_aic, best_order, best_seasonal_order, best_model_fit, seasonal_period ,X_train, y_train):
    # Set MLflow tracking URI and experiment. This ensures MLflow is initialized
    # and either connects to an existing experiment or creates a new one.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Model Training Experiment") # You might want to use the same experiment name as tuning, or a new one

    # Start an MLflow run. Using 'nested=True' as this function will likely be called
    # within a larger MLflow run. If no run is active, it will start a new top-level run.
    with mlflow.start_run(run_name="SARIMAX Model Training", nested=True):
        logger.info("Starting model training with best parameters.")

        if best_order and best_seasonal_order:
            # Original print statements retained
            print(f"\n--- Best SARIMAX Order Found: {best_order}{best_seasonal_order} with AIC: {best_aic:.2f} ---\n")
            logger.info(f"Best SARIMAX Order Found: {best_order}{best_seasonal_order} with AIC: {best_aic:.2f}")

            # Log the determined best parameters
            mlflow.log_param("final_non_seasonal_order", best_order)
            mlflow.log_param("final_seasonal_order", best_seasonal_order)
            mlflow.log_param("final_aic", best_aic)

        else:
            # Original print statements retained
            print("\nWarning: No best SARIMAX order found through grid search. Falling back to default (1,1,1)(1,1,1,s).\n")
            logger.warning("No best SARIMAX order found through grid search. Falling back to default (1,1,1)(1,1,1,s).")
            
            best_order = (1, 1, 1)
            best_seasonal_order = (1, 1, 1, seasonal_period)
            
            # Log the fallback parameters
            mlflow.log_param("final_non_seasonal_order", best_order)
            mlflow.log_param("final_seasonal_order", best_seasonal_order)
            logger.info(f"Using fallback SARIMAX order: {best_order}{best_seasonal_order}")

            model = SARIMAX(y_train, exog=X_train, order=best_order, seasonal_order=best_seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            best_model_fit = model.fit(disp=False, maxiter=100)
        
        # Original print statements retained
        print("--- Final Best SARIMAX Model Summary ---")
        print(best_model_fit.summary())
        print("\n")

        # Log the model summary as an artifact
        mlflow.log_text(str(best_model_fit.summary()), "model_summary.txt")

        # Log the fitted model itself
        # This logs the statsmodels SARIMAXResultsWrapper object
        mlflow.statsmodels.log_model(
            statsmodels_model=best_model_fit,
            artifact_path="sarimax_model",
            signature=mlflow.models.infer_signature(X_train, y_train),
            # You can add more info to the model signature if needed, e.g., input/output examples
            # input_example=X_train.head(5),
            # output_example=y_train.head(5)
        )
        logger.info("SARIMAX model trained and logged.")
        
        return best_model_fit



# making a predictions and forecasting evaluation with the metrics
def prediction_and_evaluation(best_model_fit,X_test, df , y_test, y_train):
    # Set MLflow tracking URI and experiment. This ensures MLflow is initialized
    # and either connects to an existing experiment or creates a new one.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Model Evaluation Experiment") # You might use a specific name for evaluation

    # Start an MLflow run. Using 'nested=True' as this function will likely be called
    # within a larger MLflow run. If no run is active, it will start a new top-level run.
    with mlflow.start_run(run_name="Model Prediction and Evaluation", nested=True):
        logger.info("Starting model prediction and evaluation.")

        # Predict over the test period using the best fitted model and the test exogenous variables
        # For the purpose of *only* adding mlflow and logger without changing existing code structure,
        # I'm keeping `len(y_train)` as is.
        predictions = best_model_fit.predict(
            start=y_test.index[0], # Start prediction from the beginning of the test set index
            end=y_test.index[-1],   # End prediction at the end of the test set index
            exog=X_test,
            typ='levels')

        # Assign the index from the test_data_df to predictions for easier plotting
        predictions.index = y_test.index
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        # Calculate MAPE, handling potential division by zero
        def calculate_mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0 # Avoid division by zero
            if np.sum(non_zero_mask) == 0:
                return np.nan # Or handle as appropriate if all true values are zero
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

        mape = calculate_mape(y_test, predictions)

        # Original print statements retained
        print("--- Model Evaluation Metrics ---")
        print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
        print(f"MAE (Mean Absolute Error): {mae:.2f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print("\n")

        # Log evaluation metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        
        # Log predictions as an artifact (e.g., CSV)
        # You'd typically save the predictions DataFrame to a file and then log it.
        # For instance:
        # predictions.to_csv("predictions.csv", index=True)
        # mlflow.log_artifact("predictions.csv", "evaluation_results")
        # For now, let's just log a message if we're not explicitly saving a file.
        logger.info("Model predictions generated and evaluation metrics logged.")

        return predictions




# plotting the predictions
def plot_predictions(new_predictions,  y_test , y_train, best_order, best_seasonal_order):
    # Set MLflow tracking URI and experiment. This ensures MLflow is initialized
    # and either connects to an existing experiment or creates a new one.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Model Visualization Experiment") # A dedicated experiment for plots

    # Start an MLflow run for plotting. Use 'nested=True' if this is part of a
    # larger MLflow workflow, allowing it to be a child run.
    with mlflow.start_run(run_name="Forecast Plot Generation", nested=True):
        logger.info("Starting generation and logging of forecast plot.")

        # Plot 1: Actual vs. Predicted Values
        plt.figure(figsize=(16, 8))
        plt.plot(y_train, label='Training Data', color='blue', alpha=0.7)
        plt.plot(y_test, label='Actual Test Data', color='green', marker='o', linestyle='--')
        # Using 'new_predictions' as per the function's argument
        plt.plot(new_predictions, label='SARIMAX Forecast', color='red', marker='x', linestyle='-.')
        
        plt.title(f'SARIMAX {best_order}{best_seasonal_order} Model: Actual vs. Forecasted Quantity with Lag/Time Features', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Domestic Quantity (dom_qty)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # Define the local directory to save plots
        local_plot_dir = "plots"
        os.makedirs(local_plot_dir, exist_ok=True) # Create the directory if it doesn't exist

        # Define the full path for the plot file
        plot_filename = f"forecast_plot_{best_order}_{best_seasonal_order[0:3]}.png" # Example filename
        plot_filepath = os.path.join(local_plot_dir, plot_filename)
        
        # Save the figure to the local folder instead of displaying it
        plt.savefig(plot_filepath)
        logger.info(f"Forecast plot saved locally to: {plot_filepath}")
        
        # Log the saved image as an artifact to MLflow
        mlflow.log_artifact(plot_filepath, artifact_path="forecast_visualizations")
        logger.info("Forecast plot logged to MLflow artifacts.")
        
        # Close the plot to free up memory and prevent it from being displayed by subsequent operations
        plt.close()
        


# main ()
def main():
    # Set MLflow tracking URI (if not already set, MLflow will use 'mlruns' directory by default)
    # This line is important to ensure MLflow knows where to store runs.
    # It's good practice to set it once at the top level of your script or before any runs start.
    # However, if it's already set by an outer scope, this won't re-configure it.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Data Preparation Experiment") # This will create if not exists, get if exists.

    # data importing
    df = pd.read_csv("../data/Store4128_dominicks.csv")
    
    # data prep
    new_df = data_prep(df)
    new_df.to_csv("../data/new_dom.csv")
    
    # data splitting
    X_train, y_train, X_test, y_test = data_split(new_df)
    X_train.to_csv("../data/X_train.csv")
    y_train.to_csv("../data/y_train.csv")
    X_test.to_csv("../data/X_test.csv")
    y_test.to_csv("../data/y_test.csv")
    
    # hyperparameter tuning
    best_aic, best_order, best_seasonal_order, best_model_fit, seasonal_period = hp_tuning_model(X_train, y_train)
    
    # model fitting
    model = model_training(best_aic, best_order, best_seasonal_order, best_model_fit, seasonal_period ,X_train, y_train )
    
    
    # forecasting prediction and metric evaluations
    new_predictions = prediction_and_evaluation(model, X_test, new_df, y_test , y_train)
    
    
    # plotting the final metrics
    plot_predictions(new_predictions,  y_test , y_train, best_order, best_seasonal_order)
    
    # End the main MLflow run
    mlflow.end_run()
    print("MLflow: Main run completed and ended.")

# running the entire functions code with main 
if __name__=="__main__":
    main()