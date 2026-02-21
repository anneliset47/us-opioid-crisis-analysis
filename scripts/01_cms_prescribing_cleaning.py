# Auto-exported from notebooks/01_cms_prescribing_cleaning.ipynb
# Source notebook retained for exploratory and narrative context.

# %% [markdown] cell 1
# ## Data Collection

# %% code cell 2
# request data using healthdata.gov API
import requests
import pandas as pd

url = "https://data.cms.gov/data-api/v1/dataset/c37ebe6d-f54f-4d7d-861f-fefe345554e6/data"
response = requests.get(url)
data = response.json()  # convert JSON response to a list of dictionaries
df = pd.DataFrame(data)  # create a DataFrame from the data
df.to_csv("cms_opioid_data.csv", index=False)  # save to CSV file

# %% code cell 3
#preview data
df.head()

# %% [markdown] cell 4
# ## Cleaning and Preprocessing
# ### Handling Issues/Noise in the Data (National)

# %% code cell 5
# split data into national and state level datasets
national = df[:30]
state = df[30:]
national.head()

# %% code cell 6

#clean national data

#drop redundant or irrelevant columns
national=national.drop(['Geo_Lvl','Geo_Cd','Opioid_Prscrbng_Rate_5Y_Chg','Opioid_Prscrbng_Rate_1Y_Chg','LA_Opioid_Prscrbng_Rate_5Y_Chg','LA_Opioid_Prscrbng_Rate_1Y_Chg'],axis=1)

# %% code cell 7
# only retain rows with all types of plans
national = national[national['Plan_Type'] == 'All']

# %% code cell 8
# rename columns

national.columns = ['year','geographic_level','plan','total_opioid_claims','total_claims','nat_opioid_presc_rate','long_acting_claims','nat_long_acting_rate']
national=national.drop(['plan','total_opioid_claims','total_claims','long_acting_claims','geographic_level'],axis=1)

# %% [markdown] cell 9
# ### Understanding the Data (National)

# %% code cell 10
national.dtypes

# %% code cell 11
#convert from objects to integers
national['nat_opioid_presc_rate'] = national['nat_opioid_presc_rate'].astype(str).astype(float)
national['nat_long_acting_rate'] = national['nat_long_acting_rate'].astype(str).astype(float)
national.dtypes

# %% [markdown] cell 12
# ### Basic Statisical Analysis

# %% code cell 13
print(national.describe())

import matplotlib.pyplot as plt

# Create a figure and axes for side-by-side box plots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Plot the first box plot
ax[0].boxplot(national['nat_opioid_presc_rate'])
ax[0].set_title('National Opioid Prescription Rate')
ax[0].set_ylabel('Prescription Rate')

# Plot the second box plot
ax[1].boxplot(national['nat_long_acting_rate'])
ax[1].set_title('National Long Acting Opioid Prescription Rate')
ax[1].set_ylabel('Prescription Rate')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %% [markdown] cell 14
# ### Advanced Data Understanding

# %% code cell 15
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

data_1 = national['nat_opioid_presc_rate']
data_2 = national['nat_long_acting_rate']

# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Q-Q plot for the first dataset
stats.probplot(data_1, dist="norm", plot=axes[0])
axes[0].set_title('National Prescription Rate Q-Q plot')
axes[0].set_xlabel('Theoretical quantiles')
axes[0].set_ylabel('Opioid Prescription Rate')
axes[0].grid(True)

# Q-Q plot for the second dataset
stats.probplot(data_2, dist="norm", plot=axes[1])
axes[1].set_title('National Long Acting Opioid Prescription Rate Q-Q plot')
axes[1].set_xlabel('Theoretical quantiles')
axes[1].set_ylabel('Long Acting Prescription Rate')
axes[1].grid(True)

# Adjust spacing between plots
plt.tight_layout()
plt.show()

# %% [markdown] cell 16
# ### Visualizations (National)

# %% code cell 17
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first line for opioid prescription rate
ax1.plot(national['year'], national['nat_opioid_presc_rate'], marker='o', color='blue', label='Opioid Prescription Rate')
ax1.set_xlabel("Year")
ax1.set_ylabel("Opioid Prescription Rate", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(national['year'], national['nat_long_acting_rate'], marker='o', color='green', label='Long Acting Opioid Prescription Rate')
ax2.set_ylabel("Long Acting Opioid Prescription Rate", color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title("Overall Opioid vs Long Acting Opioid Prescription Rates Over Time")
plt.show()

# %% code cell 18
# clean state data
state.head(20)

# %% code cell 19
#drop redundant or irrelevant columns
state=state.drop(['Geo_Lvl','Geo_Cd','Opioid_Prscrbng_Rate_5Y_Chg','Opioid_Prscrbng_Rate_1Y_Chg',
                        'LA_Opioid_Prscrbng_Rate_5Y_Chg','LA_Opioid_Prscrbng_Rate_1Y_Chg'],axis=1)

# %% code cell 20
state = state[state['Plan_Type'] == 'All']
state=state.drop(['Plan_Type'],axis=1)
state.head()

# %% code cell 21
state.columns = ['year','geographic_level','total_opioid_claims','total_claims','opioid_presc_rate','long_acting_claims','long_acting_rate']
state.size
print(state['year'].unique())

# %% code cell 22
state_2019 = state[state['year'] == '2019'].copy()
state_2019=state_2019.drop(['total_opioid_claims','total_claims','long_acting_claims'],axis=1)

# %% code cell 23
state_2019.head(20)

# %% code cell 24
state_2019['opioid_presc_rate'] = state_2019['opioid_presc_rate'].astype(str).astype(float)
state_2019['long_acting_rate'] = state_2019['long_acting_rate'].astype(str).astype(float)
#state['total_opioid_claims'] = state['total_opioid_claims'].astype(str).astype(float)
#state['long_acting_rate'] = state['long_acting_rate'].astype(str).astype(float)
#state['year'] = state['year'].astype(str)

# %% code cell 25
import matplotlib.pyplot as plt

# Create a figure and axes for side-by-side box plots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Plot the first box plot
ax[0].boxplot(state_2019['opioid_presc_rate'])
ax[0].set_title('Opioid Prescription Rate by State')
ax[0].set_ylabel('Prescription Rate')

# Plot the second box plot
ax[1].boxplot(state_2019['long_acting_rate'])
ax[1].set_title('Long Acting Opioid Prescription Rate by State')
ax[1].set_ylabel('Long Acting Prescription Rate')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %% code cell 26
# Get the top 10 states by opioid prescription rate
top_10_states = state_2019.nlargest(10, 'opioid_presc_rate')

# Create a horizontal bar chart with the same color scheme
fig, ax = plt.subplots(figsize=(10, 8))
colors = [gdf[gdf['NAME'] == state]['value_determined_color'].iloc[0] for state in top_10_states['geographic_level']]

# Plot the horizontal bar chart
top_10_states.plot(
    kind='barh',  # Horizontal bar chart
    x='geographic_level',
    y='opioid_presc_rate',
    color=colors,
    legend=False,
    ax=ax
)

# Add titles and labels
plt.title('Top 10 States by Opioid Prescription Rate in 2019')
plt.xlabel('Opioid Prescription Rate')
plt.ylabel('State')

# Show the plot
plt.show()

# %% code cell 27

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import missingno as msno
import os
import wget
import openpyxl
import math

# %% code cell 28
wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")

# %% code cell 29
gdf = gpd.read_file(os.getcwd()+'/cb_2018_us_state_500k.zip')
gdf.head()

# %% code cell 30
gdf = gdf.merge(state_2019,left_on='NAME',right_on='geographic_level')

# %% code cell 31
gdf.head()

# %% code cell 32
gdf.to_crs({'init':'epsg:2163'})

# %% code cell 33
# Apply this to the gdf to ensure all states are assigned colors by the same func
def makeColorColumn(gdf,variable,vmin,vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrBr)
    gdf['value_determined_color'] = gdf[variable].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
    return gdf

# %% code cell 34
# **************************
# set the value column that will be visualised
variable = 'opioid_presc_rate'

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf.opioid_presc_rate.min(), gdf.opioid_presc_rate.max() #math.ceil(gdf.pct_food_insecure.max())
# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"
gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

print(gdf.columns)
print(visframe.columns)

# %% code cell 35
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname': 'DejaVu Sans'}

# add a title and annotation
ax.set_title('Opioid Prescription Rates in 2019', **hfont, fontdict={'fontsize': '35', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

cbax.set_title('Percentage of opioid prescriptions\n', **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, \
                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
fig.colorbar(sm, cax=cbax, format=comma_fmt)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)




# create map
# Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.geographic_level not in ['Alaska','Hawaii']:
        vf = visframe[visframe.geographic_level==row.geographic_level]
        c = gdf[gdf.geographic_level==row.geographic_level][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# %% code cell 36
# Get the top 10 states by opioid prescription rate
top_10_states = state_2019.nlargest(10, 'long_acting_rate')

# Create a horizontal bar chart with the same color scheme
fig, ax = plt.subplots(figsize=(10, 8))
colors = [gdf[gdf['NAME'] == state]['value_determined_color'].iloc[0] for state in top_10_states['geographic_level']]

# Plot the horizontal bar chart
top_10_states.plot(
    kind='barh',  # Horizontal bar chart
    x='geographic_level',
    y='opioid_presc_rate',
    color=colors,
    legend=False,
    ax=ax
)

# Add titles and labels
plt.title('Top 10 States by Long Acting Opioid Prescription Rate in 2019')
plt.xlabel('Long Acting Opioid Prescription Rate')
plt.ylabel('State')

# Show the plot
plt.show()

# %% code cell 37
print(state_2019.describe())

# %% code cell 38
