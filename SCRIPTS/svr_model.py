import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load your final_data DataFrame
final_data = pd.read_csv("final_data.csv")

# Convert 'Record Date' column to datetime type
final_data['Record Date'] = pd.to_datetime(final_data['Record Date'], format='%m/%d/%y')

# Sort the data by 'Record Date'
final_data = final_data.sort_values(by='Record Date')

# Iterate over each type of security to train and test SVR models
for security in final_data['Security Description'].unique():
    security_data = final_data[final_data['Security Description'] == security]

    # Splitting data into train and test sets
    train_size = int(len(security_data) * 0.90)
    train_data, test_data = security_data.iloc[0:train_size], security_data.iloc[train_size:]

    # Train the SVR model
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Reshape input arrays to 2D
    X_train = train_data[['Record Date']].values.reshape(-1, 1)
    y_train = train_data['Average Interest Rate Amount'].values.reshape(-1, 1)
    X_test = test_data[['Record Date']].values.reshape(-1, 1)

    scaled_X_train = scaler_X.fit_transform(X_train)
    scaled_y_train = scaler_y.fit_transform(y_train)

    svr_model = SVR(kernel='rbf')
    svr_model.fit(scaled_X_train, scaled_y_train.ravel())  # ravel to convert 2D array to 1D

    # Scale the test data
    scaled_X_test = scaler_X.transform(X_test)

    # Generate predictions for SVR model
    svr_predictions_scaled = svr_model.predict(scaled_X_test)

    # Inverse scaling to get actual predictions
    svr_predictions = scaler_y.inverse_transform(svr_predictions_scaled.reshape(-1, 1)).ravel()  # reshape back to 1D

    # True test values
    true_test_values = test_data['Average Interest Rate Amount'].values

    # Calculate mean squared error for SVR
    svr_MSE_error = mean_squared_error(true_test_values, svr_predictions)
    print(f'Testing Mean Squared Error for {security} (SVR):', svr_MSE_error)

    # Visualize predicted vs. actual interest rates using SVR
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Record Date'], svr_predictions, color='orange', marker='o', linestyle='dashed', label='SVR Predicted Rates')
    plt.plot(test_data['Record Date'], true_test_values, color='red', label='Actual Rates')
    plt.title(f'{security} Interest Rate Predictions (SVR)')
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Define forecast horizon
    forecast_horizon = 60  # Forecasting interest rates for the next 60 periods

    # Train the SVR model with all available data
    X_all = security_data[['Record Date']].values.reshape(-1, 1)
    y_all = security_data['Average Interest Rate Amount'].values.reshape(-1, 1)

    scaled_X_all = scaler_X.transform(X_all)
    scaled_y_all = scaler_y.transform(y_all)

    svr_model_all = SVR(kernel='rbf')
    svr_model_all.fit(scaled_X_all, scaled_y_all.ravel())  # ravel to convert 2D array to 1D

    # Generate forecasts for SVR model
    future_dates = pd.date_range(start=security_data['Record Date'].iloc[-1], periods=forecast_horizon+1, freq='M')[1:]  # Exclude the current date
    scaled_future_dates = scaler_X.transform(future_dates.values.reshape(-1, 1))
    svr_forecasts_scaled = svr_model_all.predict(scaled_future_dates)
    svr_forecasts = scaler_y.inverse_transform(svr_forecasts_scaled.reshape(-1, 1)).ravel()  # reshape back to 1D

    # Visualize forecasted interest rates using SVR
    plt.figure(figsize=(10, 6))
    plt.plot(security_data['Record Date'], security_data['Average Interest Rate Amount'], color='blue', label='Historical Rates')
    plt.plot(future_dates, svr_forecasts, color='orange', linestyle='dashed', label='SVR Forecasted Rates')
    plt.title(f'{security} Interest Rate Forecast (SVR)')
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

