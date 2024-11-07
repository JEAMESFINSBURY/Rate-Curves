# %%
import pandas as pd
import requests 
import io
import numpy as np
import datetime as dt
import pprint as pp
import logging
import json
import time
import matplotlib.pyplot as plt
# Define URL Endpoint
Bank_of_England = 'https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes'

scenario_launch_date = '2023-09-30'
scenario_launch_date = pd.to_datetime(scenario_launch_date)
scenario_launch_date_ONS = dt.datetime.strftime(scenario_launch_date,"%Y-%m")


Base_Rate = {
    'Datefrom'   : '01/Jan/1970',
    'SeriesCodes': 'IUDSOIA',
    'CSVF'       : 'TN',
    'UsingCodes' : 'Y',
    'VPD'        : 'Y',
    'VFD'        : 'N'
}

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/54.0.2840.90 '
                  'Safari/537.36'
}

Baserate = requests.get(Bank_of_England, params=Base_Rate, headers=headers)

# # Check if the response was successful, it should return '200'
print(Baserate.status_code)
print(Baserate.url)



# Ingest Base Rate & Format Data Set
df_Base_Rate = pd.read_csv(io.BytesIO(Baserate.content))
df_Base_Rate['DATE'] = pd.to_datetime(df_Base_Rate['DATE'])
df_Base_Rate = df_Base_Rate.rename(columns={'DATE':'Date','IUDSOIA': 'ON_Rate'})
df_Base_Rate

plt.figure(figsize=(12, 6))
plt.plot(df_Base_Rate['Date'], df_Base_Rate['ON_Rate'], label='Historical O/N Rate')

plt.title('O/N Rate Historical UK')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()