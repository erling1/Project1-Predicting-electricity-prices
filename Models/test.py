import requests
import json
import os
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Safely access the API key from environment variables
env_vars = dotenv_values('strømpriser_api_key.env')

api_key = env_vars['MY_API_KEY']

date1 = date(2022, 8, 15)
date2 = date(2024, 8, 15)

timespan = (date1.isoformat(), date2.isoformat())

def data_from_api(timespan: tuple , region: int, datafile: str):

    base_url =  "https://api.strompriser.no/public/prices"

    startDate, endDate = timespan

    params = {  "region": region,
                "startDate": startDate,
                "endDate" : endDate}

    headers = {"api-key" :  api_key}

    response = requests.get(base_url, params=params, headers=headers)

    if response.ok:
    
        data = response.json()  # Parse the response as JSON

        with open(datafile, 'w') as json_file:
            json.dump(data, json_file)
    else:
        # If the response is not OK (e.g., 400, 404, 500)
        print(f"Failed to retrieve data. Status code: {response.status_code}")



df = pd.read_json('strømpriser_data.json')
df_dailyPrice_date = df[['dailyPriceArray','date']]

#Create one new row for each item in dailyPriceArray
df_exploded = df_dailyPrice_date.explode('dailyPriceArray').reset_index(drop=True)
df_exploded['hour'] = df_exploded.groupby('date').cumcount()
df_exploded = df_exploded.rename(columns={'dailyPriceArray': 'price'})


#Pivot the DataFrame to have 'hour' as index, and one column per 'date'
pivot_df__dailyPrice_date = df_exploded.pivot(index='hour', columns='date', values='price')

#Drow last row
pivot_df__dailyPrice_date = pivot_df__dailyPrice_date.iloc[:-1]


melted_df = pivot_df__dailyPrice_date.reset_index().melt(id_vars='hour', var_name='date', value_name='price')

# Assuming melted_df is correctly formatted
for date in melted_df['date'].unique():
    subset = melted_df[melted_df['date'] == date]
    plt.plot(subset['hour'], subset['price'],  label=date, linewidth=0.5)

plt.title('Hourly Prices per Date')
plt.xlabel('Hour of Day')
plt.ylabel('Price')
plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Get the numeric values of dates for color mapping
melted_df['date_num'] = melted_df['date'].map(pd.Timestamp.timestamp)

# Normalize the date_num values to be in the range [0, 1]
norm = plt.Normalize(melted_df['date_num'].min(), melted_df['date_num'].max())
melted_df['color'] = [plt.cm.Blues(norm(val)) for val in melted_df['date_num']]

# Plotting
plt.figure(figsize=(10, 6))

# Plot with color mapping
for date in melted_df['date'].unique():
    subset = melted_df[melted_df['date'] == date]
    plt.plot(subset['hour'], subset['price'], marker='o', label=date, color=subset['color'].iloc[0], linewidth=0.5)

plt.title('Hourly Prices per Date with Color Gradient')
plt.xlabel('Hour of Day')
plt.ylabel('Price')
plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print(pivot_df__dailyPrice_date)
# Ensure all values are numeric
pivot_df__dailyPrice_date = pivot_df__dailyPrice_date.apply(pd.to_numeric, errors='coerce')

# Calculate the indices for 5 equally spaced dates
num_dates = len(pivot_df__dailyPrice_date.columns)
date_indices = np.linspace(0, num_dates - 1, num=5, dtype=int)


plt.figure(figsize=(12, 6))
plt.imshow(pivot_df__dailyPrice_date, aspect='auto', cmap='Blues', interpolation='nearest')

# Customize the ticks and labels
plt.colorbar(label='Price')  # Add a color bar to indicate price
plt.title('Hourly Prices Heatmap')
plt.xlabel('Date')
plt.ylabel('Hour of Day')
plt.xticks(ticks=date_indices, labels=pivot_df__dailyPrice_date.columns[date_indices].strftime('%Y-%m-%d'), rotation=45)
plt.yticks(ticks=np.arange(len(pivot_df__dailyPrice_date.index)), labels=pivot_df__dailyPrice_date.index)

# Show the heatmap
plt.tight_layout()
plt.show()