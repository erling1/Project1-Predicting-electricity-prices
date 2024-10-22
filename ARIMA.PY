
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
from creatingdataset import dataset_ele_prices
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import statsmodels.api as sm

dataframe = dataset_ele_prices(startDate=date(2019,10,10),endDate=date(2013,10,10), filename="Strømpriser_for_ARIMA.json")

df = dataframe[['date','dailyPriceAverage']]

#Testing for Stationarity
adf_test = adfuller(df['dailyPriceAverage'])
# Output the results
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

plt.figure()
plot_acf(df['dailyPriceAverage'], lags=40)
plt.savefig('Autocorrelation.png')
plt.figure()
plot_pacf(df['dailyPriceAverage'], lags=40)
plt.savefig('partial Autocorrelation.png')

#Plots suggest ARIMA(1, 0, 0) model is the most optimal

"""model = ARIMA(df['dailyPriceAverage'], order=(1, 0, 0))
fit_model = model.fit()"""

#forecast = fit_model.get_forecast(steps=30)


train_size = int(len(df['dailyPriceAverage'])*0.8)
train_set, test_set = df['dailyPriceAverage'][:train_size], df['dailyPriceAverage'][train_size:] 

#Training the Arima model
model_train = ARIMA(train_set, order=(1, 0, 0))
model_train_fit = model_train.fit()

# Forecast the trained model on the test dataset
test_forecast = model_train_fit.get_forecast(steps=len(test_set))
test_forecast_series = pd.Series(test_forecast.predicted_mean, index=test_set.index)

# Create a plot to compare the forecast with the actual test data
plt.figure(figsize=(14,7))
plt.plot(train_set, label='Training Data')
plt.plot(test_set, label='Test Data', color='orange')
plt.plot(test_forecast_series, label='Forecasted Data', color='green')
plt.fill_between(test_set.index, 
                 test_forecast.conf_int().iloc[:, 0], 
                 test_forecast.conf_int().iloc[:, 1], 
                 color='k', alpha=.15)
plt.title('ARIMA Model Evaluation')
plt.xlabel('Date')
plt.ylabel('Avg Electricity Price')
plt.legend()
plt.savefig('ARIMA Model.png')


#model = auto_arima(df['dailyPriceAverage'], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
#print(model.summary())

#Best model: ARIMA(2,1,1)(0,0,1)

sarima_model = sm.tsa.SARIMAX(train_set, 
                               order=(2, 1, 1), 
                               seasonal_order=(0, 0, 1, 300))
sarima_fit = sarima_model.fit()

# Forecast the trained model on the test dataset
test_forecast_sarima = sarima_fit.get_forecast(steps=len(test_set))
test_forecast_series_sarima = pd.Series(test_forecast_sarima.predicted_mean, index=test_set.index)

# Create a plot to compare the forecast with the actual test data
plt.figure(figsize=(14,7))
plt.plot(train_set, label='Training Data')
plt.plot(test_set, label='Test Data', color='orange')
plt.plot(test_forecast_series_sarima, label='Forecasted Data', color='green')
plt.fill_between(test_set.index, 
                 test_forecast.conf_int().iloc[:, 0], 
                 test_forecast.conf_int().iloc[:, 1], 
                 color='k', alpha=.15)
plt.title('ARIMA Model Evaluation')
plt.xlabel('Date')
plt.ylabel('Avg Electricity Price')
plt.legend()
plt.savefig('SARIMA Model.png')