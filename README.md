# 📈 Time Series Forecasting with Azure Machine Learning

This project is part of the **DP-100 Certification** and focuses on developing and deploying a robust **time series forecasting** solution using **Azure Machine Learning**.

---

## 🧠 Project Summary

- Developed a machine learning pipeline for forecasting sales data using the **OJ Sales Simulated dataset** from Azure Open Datasets.
- The original dataset contained over **12,000 CSV files** representing 3 stores for each date.
- Initial exploration and aggregation were done to unify the dataset and prepare it for training.
- Preprocessed data to include **lag features**, **date components (day, month, year)**, and handled **multiple time series identifiers**.
- Managed compute clusters and resources in Azure ML for efficient model training and testing.

---

## ⚙️ Modeling Process

- Attempted **automated ML in Azure** with time series forecasting mode. Faced issues due to multi-store data.
- Filtered the data to a single store (`Dominicks 4128`) for more accurate modeling.
- Top-performing models from AutoML included **ARIMA**, **Seasonality**, and **Moving Average**.
- Custom model built using **SARIMAX** (from `statsmodels`) with extensive **hyperparameter tuning**.

---

## 🧪 Evaluation Metrics

| Metric                | Value     |
|----------------------|-----------|
| MAPE                 | 28.78     |
| MAE                  | 3299.61   |
| RMSE                 | 4617.57   |
| Spearman Correlation | 1.0       |
| R2 Score             | 0.971     |

---

## 🧩 Modular Coding & MLFlow Integration

- Implemented **function-based modular coding** using `main.py` in Spyder IDE.
- Integrated **MLflow** for tracking hyperparameters, metrics, and artifacts.
- Included **logging**, **exception handling**, and **visualizations** (residuals, subplots, etc.).

---

## ☁️ Azure Deployment: Batch Endpoint

- Registered the model and deployed it via an **offline batch endpoint** on Azure.
- Used `azureml` and `azureaiml` libraries for workspace authentication and job submission.
- Created compute instance and batch endpoint using `BatchEndpoint` and `BatchDeployment`.
- Deployed the MLflow model with configurations like:
  - `instance_count`, `mini_batch_size`, `output_action`, `retry_settings`, etc.
- Monitored endpoint progress in **Azure ML Studio** and downloaded predictions from `predictions.csv`.

---

## 📤 Output & Results

- All inference results were logged and stored in Azure's default datastore.
- Final predictions visualized using Pandas and Matplotlib.

---

## 🔗 Key Technologies

- Azure ML SDK
- Python (Pandas, Numpy, Statsmodels)
- MLflow
- AutoML
- SARIMAX
- Batch Endpoint
- Logging & Modular Code

---

## 🙌 Acknowledgments

This project was built under the **SkillUpOnline DP-100 Certification Program**. Special thanks to the mentors and community that supported this journey.

---

## 🔗 Let’s Connect

If you're interested in collaborating or have feedback:

[📎 Connect on LinkedIn](https://www.linkedin.com/in/ashwin-ashok-2826a3173/)

