#%% Import Required Packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fredapi import Fred
sns.set_style("white")
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Plotly
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from scipy.optimize import minimize

### Define the colour scheme
c1 = "#173f5f"
c2 = "#20639b"
c3 = "#3caea3"
c4 = "#f6d55c"
c5 = "#ed553b"

custom_palette = [c1, c2, c3, c4, c5]
sns.palplot(sns.color_palette(custom_palette))

OBSERVATION_START = '1970-06-01'

#%% Import UK BOE POLICY RATES RATES From FEDERAL RESERVE API

fred = Fred(api_key='3c21aa696b7412a91e864e5691081d92')

rates = fred.get_series('BOERUKQ', observation_start='1975-01-01')
df = pd.DataFrame(rates, columns=['rate'])
df.index.name = 'date'

# Calculate daily returns and remove any NaN values
df['rate_diff'] = df['rate'].diff()
df = df.dropna()

# Basic statistics
print("Data Summary:")
print(df.describe())

# Plot historical rates
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['rate'])
plt.title('Bank of England Base Rate Historical Data')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.grid(True)
plt.show()

# %%

