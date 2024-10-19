import os
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone, date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import  requests
import json





def data_from_api(timespan: tuple , region: int, datafile: str, api_key: str):

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