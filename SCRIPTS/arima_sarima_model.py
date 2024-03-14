# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load your final_data DataFrame
final_data = pd.read_csv("final_data.csv")

# Convert 'Record Date' column to datetime type
final_data['Record Date'] = pd.to_datetime(final_data['Record Date'], format='%m/%d/%y')

# Sort the data by 'Record Date'
final_data = final_data.sort_values(by='Record Date')

# Iterate over each type of security to train and test ARIMA and SARIMA models
for security in final_data['Security Description'].unique():
    security_data = final_data[final_data['Security Description'] == security]

    # Splitting data into train and test sets
    train_size = int(len(security_data) * 0.90)
    train_data, test_data = security_data.iloc[0:train_size], security_data.iloc[train_size:]

    # Define ARIMA parameters
    p = 3  # Number of lag observations
    d = 0  # Number of differences
    q = 0  # Size of the moving average window

    # Train the ARIMA model
    arima_model = ARIMA(train_data['Average Interest Rate Amount'], order=(p, d, q))
    arima_model_fit = arima_model.fit()

    # Define SARIMA parameters
    order = (3, 0, 0)  # ARIMA order (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, S)

    # Train the SARIMA model
    sarima_model = SARIMAX(train_data['Average Interest Rate Amount'], order=order, seasonal_order=seasonal_order)
    sarima_model_fit = sarima_model.fit()

    # Generate predictions for ARIMA model
    arima_predictions = arima_model_fit.forecast(steps=len(test_data))

    # Generate predictions for SARIMA model
    sarima_predictions = sarima_model_fit.forecast(steps=len(test_data))

    # True test values
    true_test_values = test_data['Average Interest Rate Amount'].values

    # Calculate mean squared error for ARIMA
    arima_MSE_error = mean_squared_error(true_test_values, arima_predictions)
    print(f'Testing Mean Squared Error for {security} (ARIMA):', arima_MSE_error)

    # Calculate mean squared error for SARIMA
    sarima_MSE_error = mean_squared_error(true_test_values, sarima_predictions)
    print(f'Testing Mean Squared Error for {security} (SARIMA):', sarima_MSE_error)

    # Visualize predicted vs. actual interest rates
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Record Date'], arima_predictions, color='blue', marker='o', linestyle='dashed', label='ARIMA Predicted Rates')
    plt.plot(test_data['Record Date'], sarima_predictions, color='green', marker='o', linestyle='dashed', label='SARIMA Predicted Rates')
    plt.plot(test_data['Record Date'], true_test_values, color='red', label='Actual Rates')
    plt.title(f'{security} Interest Rate Predictions (ARIMA vs SARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


# Iterate over each type of security
for security in final_data['Security Description'].unique():
    security_data = final_data[final_data['Security Description'] == security]

    # Define ARIMA parameters
    p = 3  # Number of lag observations
    d = 0  # Number of differences
    q = 0  # Size of the moving average window

    # Train the ARIMA model
    arima_model = ARIMA(security_data['Average Interest Rate Amount'], order=(p, d, q))
    arima_model_fit = arima_model.fit()

    # Define SARIMA parameters
    order = (3, 0, 0)  # ARIMA order (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, S)

    # Train the SARIMA model
    sarima_model = SARIMAX(security_data['Average Interest Rate Amount'], order=order, seasonal_order=seasonal_order)
    sarima_model_fit = sarima_model.fit()

    # Define the number of future periods to forecast
    forecast_horizon = 60  # Forecasting interest rates for the next 12 months

    # Generate forecasts for ARIMA model
    arima_forecast = arima_model_fit.forecast(steps=forecast_horizon)

    # Generate forecasts for SARIMA model
    sarima_forecast = sarima_model_fit.forecast(steps=forecast_horizon)

    # Visualize forecasted interest rates
    plt.figure(figsize=(10, 6))
    plt.plot(security_data['Record Date'], security_data['Average Interest Rate Amount'], color='blue', label='Historical Rates')
    plt.plot(pd.date_range(start=security_data['Record Date'].iloc[0], periods=len(security_data)+forecast_horizon, freq='M'), np.concatenate([security_data['Average Interest Rate Amount'].values, arima_forecast]), color='green', linestyle='dashed', label='ARIMA Forecasted Rates')
    plt.plot(pd.date_range(start=security_data['Record Date'].iloc[0], periods=len(security_data)+forecast_horizon, freq='M'), np.concatenate([security_data['Average Interest Rate Amount'].values, sarima_forecast]), color='purple', linestyle='dashed', label='SARIMA Forecasted Rates')
    plt.title(f'{security} Interest Rate Forecast (ARIMA vs SARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
