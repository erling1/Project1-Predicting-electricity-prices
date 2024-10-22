

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import numpy as np
from API_func import data_from_api_ele

#create json data
# Safely access the API key from environment variables
env_vars = dotenv_values('.env')

api_key = env_vars['MY_API_KEY']

date1 = date(2019, 6, 1)
date2 = date(2024, 8, 15)

timespan = (date1, date2)

data_from_api_ele(timespan=timespan, region=1, api_key=api_key, datafile="strømpriser_data_MLPRegressor.json")

df = pd.read_json('strømpriser_data_MLPRegressor.json')

df = df[['dailyPriceAverage','date']]
df['date'] = df['date'].dt.date
df['date'] = pd.to_datetime(df['date'])

df['price_lag_1'] = df['dailyPriceAverage'].shift(1)
df['price_lag_2'] = df['dailyPriceAverage'].shift(2)
df['price_lag_3'] = df['dailyPriceAverage'].shift(3)


plt.plot(df['date'], df['dailyPriceAverage'])
plt.title('Daily Price Average Over Time')
plt.savefig('price_plot.png', bbox_inches='tight')

# Split the data into training and testing sets

y_train = df[df['date'] < '2023-01-01']['dailyPriceAverage']



X_train = df[df['date'] < '2023-01-01'].drop(columns='dailyPriceAverage')
X_test = df[df['date'] > '2023-01-01'].drop(columns='dailyPriceAverage')

X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month
X_train['day'] = X_train['date'].dt.day

X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day

# Drop the original date column (you can also use it as a timestamp if needed)
X_train.drop(columns='date', inplace=True)
X_test.drop(columns='date', inplace=True)
X_train.fillna(0, inplace=True)

print(X_train)
print(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', alpha=0.01, max_iter=1000, learning_rate_init=0.01 )
model.fit(X_train_scaled,y_train)

y_test = df[df['date'] >= '2023-01-01']

# Making predictions
y_pred = model.predict(X_test_scaled)

# Check the shape of y_pred
print(f'y_pred shape: {y_pred.shape}')  # Should be (590,)

print(y_pred)

actual_daily_avg = df[df['date'] >= '2023-01-01']['dailyPriceAverage']

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot the actual prices
plt.plot(actual_daily_avg.index, actual_daily_avg.values, label='Actual Prices', color='blue')

# Plot the predicted prices
plt.plot(actual_daily_avg.index[:-1], y_pred, label='Predicted Prices', color='orange')

# Add titles and labels
plt.title('Actual vs Predicted Daily Average Prices')
plt.xlabel('Date')
plt.ylabel('Daily Average Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid()

# Show the plot
plt.tight_layout()
plt.savefig("Actual vs Predicted Daily Average Prices.png")