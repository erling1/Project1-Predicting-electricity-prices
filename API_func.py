import os
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import  requests
import json





def data_from_api_ele(timespan: tuple , region: int, datafile: str, api_key: str):

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


def weather_api(timespan: str , datafile: str):


    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    #SN90450
    params = {"sources":  'SN18700',
              "referencetime": timespan,
              "elements": "air_temperature",
              "timeoffsets": "default",  
              "levels": "default",       
              "qualities": "0,1,2,3,4" }
    
    env_vars = dotenv_values('.env')

     
    client_id = env_vars['client-id']
    client_secret = env_vars['client-secret']
    
    
    # Issue an HTTP GET request
    r = requests.get(endpoint, params, auth=(client_id,''))
    # Extract JSON data
    if r.ok:
        data = r.json()

        with open(datafile, 'w') as json_file:
                json.dump(data, json_file)

    else:
        # If the response is not OK (e.g., 400, 404, 500)
        print(f"Failed to retrieve data. Status code: {r.status_code}")

         


date1 = date(2019, 6, 1)
date2 = date(2024, 8, 15)
timespan = "2024-6-1/2024-8-15"
weather_api(timespan=timespan, datafile="air_temp")

        

    

