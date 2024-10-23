from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import numpy as np
from API_func import data_from_api_ele

def dataset_ele_prices(startDate: datetime, endDate: datetime, filename: str):
   
    #Safely access the API key from environment variables
    env_vars = dotenv_values('.env')

    api_key = env_vars['MY_API_KEY']

    timespan = (startDate, endDate)
    #Create json data, region=1: Oslo

    data_from_api_ele(timespan=timespan, region=1, api_key=api_key, datafile=filename)
    dataframe = pd.read_json('strømpriser_data_MLPRegressor.json')

    return dataframe



