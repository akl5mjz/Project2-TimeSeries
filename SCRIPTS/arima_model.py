import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the data
data = pd.read_csv("final_data v2.csv", names=['Date', 'Treasury Bills', 'Treasury Notes', 'Treasury Bonds'], header=0)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 2: Visualize the data
data.plot(figsize=(12, 8))
plt.title('Interest Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.show()

# Step 3: Train-test split
train_size = int(len(data) * 0.7)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Step 4: Define ARIMA model parameters
p = 4  # AR parameter
d = 1  # Differencing parameter
q = 0  # MA parameter

# Step 5: Initialize lists for predictions and history
history_t_bill = [x for x in train_data['Treasury Bills']]
history_t_note = [x for x in train_data['Treasury Notes']]
history_t_bond = [x for x in train_data['Treasury Bonds']]

model_predictions_t_bill = []
model_predictions_t_note = []
model_predictions_t_bond = []

# Step 6: Fit ARIMA model and make predictions for T-Bills
for time_point in range(len(test_data)):
    model_t_bill = ARIMA(history_t_bill, order=(p, d, q))
    model_fit_t_bill = model_t_bill.fit()
    output_t_bill = model_fit_t_bill.forecast()
    yhat_t_bill = output_t_bill[0]
    model_predictions_t_bill.append(yhat_t_bill)
    true_test_value_t_bill = test_data['Treasury Bills'].iloc[time_point]
    history_t_bill.append(true_test_value_t_bill)

# Step 7: Fit ARIMA model and make predictions for T-Notes
for time_point in range(len(test_data)):
    model_t_note = ARIMA(history_t_note, order=(p, d, q))
    model_fit_t_note = model_t_note.fit()
    output_t_note = model_fit_t_note.forecast()
    yhat_t_note = output_t_note[0]
    model_predictions_t_note.append(yhat_t_note)
    true_test_value_t_note = test_data['Treasury Notes'].iloc[time_point]
    history_t_note.append(true_test_value_t_note)

# Step 8: Fit ARIMA model and make predictions for T-Bonds
for time_point in range(len(test_data)):
    model_t_bond = ARIMA(history_t_bond, order=(p, d, q))
    model_fit_t_bond = model_t_bond.fit()
    output_t_bond = model_fit_t_bond.forecast()
    yhat_t_bond = output_t_bond[0]
    model_predictions_t_bond.append(yhat_t_bond)
    true_test_value_t_bond = test_data['Treasury Bonds'].iloc[time_point]
    history_t_bond.append(true_test_value_t_bond)

# Step 9: Calculate Mean Squared Error for each series
MSE_error_t_bill = mean_squared_error(test_data['Treasury Bills'], model_predictions_t_bill)
MSE_error_t_note = mean_squared_error(test_data['Treasury Notes'], model_predictions_t_note)
MSE_error_t_bond = mean_squared_error(test_data['Treasury Bonds'], model_predictions_t_bond)

print('T-Bill Testing Mean Squared Error is {}'.format(MSE_error_t_bill))
print('T-Note Testing Mean Squared Error is {}'.format(MSE_error_t_note))
print('T-Bond Testing Mean Squared Error is {}'.format(MSE_error_t_bond))

# Step 10: Plot predictions vs actual values for each series
plt.figure(figsize=(20,10))

# Plot T-Bill predictions
plt.subplot(3, 1, 1)
plt.plot(test_data.index, model_predictions_t_bill, color='blue', marker='o', linestyle='dashed',label='Predicted T-Bill Rate')
plt.plot(test_data.index, test_data['Treasury Bills'], color='red', label='Actual T-Bill Rate')
plt.title('T-Bill Interest Rate Prediction')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.legend()

# Plot T-Note predictions
plt.subplot(3, 1, 2)
plt.plot(test_data.index, model_predictions_t_note, color='green', marker='o', linestyle='dashed',label='Predicted T-Note Rate')
plt.plot(test_data.index, test_data['Treasury Notes'], color='purple', label='Actual T-Note Rate')
plt.title('T-Note Interest Rate Prediction')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.legend()

# Plot T-Bond predictions
plt.subplot(3, 1, 3)
plt.plot(test_data.index, model_predictions_t_bond, color='orange', marker='o', linestyle='dashed',label='Predicted T-Bond Rate')
plt.plot(test_data.index, test_data['Treasury Bonds'], color='brown', label='Actual T-Bond Rate')
plt.title('T-Bond Interest Rate Prediction')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.legend()

plt.tight_layout()
plt.show()


# Step 1: Load the Data
data = pd.read_csv("final_data v2.csv")

# Step 2: Preprocess the Data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.asfreq('M')  # Ensure monthly frequency

# Step 3: Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Step 4: Model Specification and Parameter Tuning
best_orders = {}
best_mse = {}

# Define a range of values for p, d, q
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

for column in train.columns:
    best_order = None
    best_mse[column] = np.inf
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train[column], order=(p, d, q))
                    fit_model = model.fit()
                    forecast = fit_model.forecast(steps=len(test))
                    mse = mean_squared_error(test[column], forecast)
                    if mse < best_mse[column]:
                        best_order = (p, d, q)
                        best_mse[column] = mse
                except:
                    continue
    best_orders[column] = best_order

# Step 5: Fit the Final Models
final_models = {}
for column in train.columns:
    final_model = ARIMA(data[column], order=best_orders[column])
    final_fit_model = final_model.fit()
    final_models[column] = final_fit_model

# Step 6: Forecast
forecasts = {}
for column in train.columns:
    forecasts[column] = final_models[column].forecast(steps=120)  # Forecasting 10 years (120 months) ahead

# Step 7: Visualize the Forecasts
plt.figure(figsize=(10, 6))
for column in data.columns:
    plt.plot(data.index, data[column], label=f'Actual {column}')
    plt.plot(pd.date_range(start=train.index[-1], periods=len(test)+len(forecasts[column]), freq='M'), np.concatenate([test[column], forecasts[column]]), label=f'Forecasted {column}')
plt.title('Forecasted Interest Rates')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.legend()
plt.show()

