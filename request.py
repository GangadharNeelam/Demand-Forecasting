import requests
import pandas as pd
import json
from datetime import datetime


url = 'http://localhost:5000/api'
#Number of months we need to forecast
data = {'number_of_months' : 3}

#sent the input to the flask app
r = requests.post(url, json=data)

#Load the output using json
output = json.loads(r.json())

#Convert the output into data frame
output_df = pd.DataFrame(output)

#Covert to seconds
output_df["Date"] = pd.to_numeric(output_df["Date"])/1000
#Convert into date
output_df["Date"] = pd.to_datetime(output_df["Date"], unit="s")
#Give the date format
output_df["Date"] = output_df["Date"].dt.strftime("%d-%m-%Y")
print(output_df)