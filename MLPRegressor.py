

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import numpy as np
from API_func import data_from_api

#create json data
# Safely access the API key from environment variables
env_vars = dotenv_values('strømpriser_api_key.env')

api_key = env_vars['MY_API_KEY']

date1 = date(2019, 6, 1)
date2 = date(2024, 8, 15)

timespan = (date1, date2)

data_from_api(timespan=timespan, region=1, api_key=api_key, datafile="strømpriser_data_MLPRegressor.json")

df = pd.read_json('strømpriser_data_MLPRegressor.json')
df = df[['dailyPriceAverage','date']]
df['date'] = df['date'].dt.date
df['date'] = pd.to_datetime(df['date'])

df['price_lag_1'] = df['dailyPriceAverage'].shift(1)
df['price_lag_2'] = df['dailyPriceAverage'].shift(2)
df['price_lag_3'] = df['dailyPriceAverage'].shift(3)

print(df)

plt.plot(df['date'], df['dailyPriceAverage'])
plt.title('Daily Price Average Over Time')
plt.savefig('price_plot.png', bbox_inches='tight')

# Split the data into training and testing sets

y_train = df[df['date'] < '2023-01-01']['dailyPriceAverage']
X_train = df[df['date'] < '2023-01-01']
print(y_train)

train(X,y)

y_test = df[df['date'] >= '2023-01-01']



