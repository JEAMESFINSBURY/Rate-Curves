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
import sklearn as sk
import sklearn.preprocessing as sk_pre
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

Tenors = ['1-Month','3-Month','1-Year','2-Year','5-Year','10-Year']
TS = [1,2]

TS_Start = '1995-01-05'
TS_End = '2024-11-01'
#  Ingest Yield Curve Data --> Source: Investor.com Historical Data downloads

Yield_Curve_DF_Master = pd.DataFrame({'Date': pd.date_range(TS_Start,TS_End, freq='D')})

for T in Tenors:

    Yield_Curve_df = pd.read_csv('United Kingdom {} Bond Yield Historical Data (1).csv'.format(T))
    Yield_Curve_df = Yield_Curve_df[['Date','Price']]
    Yield_Curve_df['Rate_{}'.format(T)] = Yield_Curve_df['Price']
    Yield_Curve_df['Date'] = pd.to_datetime(Yield_Curve_df['Date'])

    Yield_Curve_df = Yield_Curve_df.drop(['Price'], axis=1)
    Yield_Curve_DF_Master = Yield_Curve_DF_Master.merge(Yield_Curve_df, how='left', on='Date')


Yield_Curve_DF_Master.set_index(Yield_Curve_DF_Master['Date'], drop=True, inplace=True)
Yield_Curve_DF_Master = Yield_Curve_DF_Master.drop(['Date'], axis=1)
Yield_Curve_DF_Master = Yield_Curve_DF_Master.interpolate()

# %% Summary of Details
Yield_Curve_DF_Master.describe()
Yield_Curve_DF_Master.to_csv('Yieldfinal.csv')


#%% Plot Function
def plot_rates(df):
    plot_vars = df.columns
    
    fig, ax = plt.subplots(figsize=(8,4), ncols=1, nrows=1)
    for var in plot_vars:
        ax.plot(df.index, df[var], label=var, lw=1.0)
    ax.set(title='Daily Interest Rate', xlabel='Date', ylabel='Interest Rate (%)')
    fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    plt.show()

# %%Yield Curves Chart
Yield_Curves = Yield_Curve_DF_Master.copy()
plot_rates(Yield_Curves)
# %% Standardize the Time Series Data

scaler = sk_pre.StandardScaler()
Yield_Curves_scaled = pd.DataFrame(scaler.fit_transform(np.array(Yield_Curves)))
Yield_Curves_scaled.columns = Yield_Curves.columns
Yield_Curves_scaled.to_csv('Scaled.csv')
plot_rates(Yield_Curves_scaled)

# Compute Covariance Matrix Between Yield Curves

covariance_matrix = Yield_Curves_scaled.cov()

fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
ax.set(title='Covariance Matrix (Standardized Yield Values)')

# %% Deconstruct EigenValues

eig_vals, eig_vect = np.linalg.eig(covariance_matrix)

print(eig_vals[0])
print(eig_vect[0])
# %% Order the EigenVectors to determine the magnitude

idx = np.argsort(eig_vals, axis=0) # Sorting the EigenValues(Scalars) in magnitude (ascending order) - outputs are row indices
idx_desc = np.argsort(eig_vals,axis=0)[::-1] # Sorting Scalars in Descending Order

sorted_eigenvectors = eig_vect[:, idx_desc]

# Cumulative Sum of the Eigen Values (Scalars)
cumsum_eig_values = np.cumsum(eig_vals[idx_desc]) / np.sum(eig_vals[idx_desc])
xint = range(1,len(cumsum_eig_values)+1)

fig, ax = plt.subplots(figsize=(12,4), ncols=2, nrows=1)
ax[0].plot(xint, cumsum_eig_values)
ax[0].set(title='Principle Components', xlabel='Number of Components', ylabel='Cumulative Explainability')

ax[1].plot(xint[1:], cumsum_eig_values[1:])
ax[1].set(title='Principle Components', xlabel='Number of Components', ylabel='Cumulative Explainability')
plt.show()

# %% Determining the impact of multiplying the two main principal component scores with the Scaled Yield Values through time for each Yield Curve Tenor

pc_scores = np.dot(Yield_Curves_scaled, sorted_eigenvectors[:, :2])

fig, ax = plt.subplots(figsize=(6,4), ncols=1, nrows=1)
for i, pca in enumerate(['level','slope']):
    ax.plot(pc_scores[:,i], label=pca)
ax.set(title='Principle Components', xlabel='Date', ylabel='Eigen Scores')
fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
plt.show()

df_pca = pd.DataFrame(pc_scores, columns=['PC1','PC2'])
df_pca.index = Yield_Curve_DF_Master.index

print(df_pca.tail(10))

# %%
