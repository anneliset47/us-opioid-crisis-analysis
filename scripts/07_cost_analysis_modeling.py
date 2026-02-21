# Auto-exported from notebooks/07_cost_analysis_modeling.ipynb
# Source notebook retained for exploratory and narrative context.

# %% code cell 1
import requests
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% code cell 2
url = "https://www.cdc.gov/mmwr/volumes/70/wr/mm7015a1.htm"
response = requests.get(url)

# %% code cell 3
response

# %% code cell 4
content = response.content

# %% code cell 5
soup = BeautifulSoup(content, "html.parser")

# %% code cell 6
tables = soup.find_all("table")

# %% code cell 7
#first table as data frame
columns = []
for th in tables[0].find_all('th'):
  columns.append(th.get_text(strip=True))
rows = []
for tr in tables[0].find_all('tr'):
  entries = tr.find_all('td')
  row = [entry.get_text(strip=True) for entry in entries]
  rows.append(row)
df = pd.DataFrame(rows, columns=columns)
df.head()

# %% code cell 8
df.drop(0)
print()

# %% code cell 9
df.info()

# %% code cell 10
#data cleaning here to translate objects to ints as reading them in caused an issue here
df = df.rename(columns={"Jurisdiction†":"Jurisdiction"})
df['Estimated case count of opioid use disorder'] = df['Estimated case count of opioid use disorder'].replace({r'[^\d.]': ''}, regex=True)
df['Estimated case count of opioid use disorder'] = pd.to_numeric(df['Estimated case count of opioid use disorder'], errors='coerce')
df['Case count of fatal opioid overdose'] = df['Case count of fatal opioid overdose'].replace({r'[^\d.]': ''}, regex=True)
df['Case count of fatal opioid overdose'] = pd.to_numeric(df['Case count of fatal opioid overdose'], errors='coerce')
df['Cost of opioid use disorder, $ (millions)'] = df['Cost of opioid use disorder, $ (millions)'].replace({r'[^\d.]': ''}, regex=True)
df['Cost of opioid use disorder, $ (millions)'] = pd.to_numeric(df['Cost of opioid use disorder, $ (millions)'], errors='coerce')
df['Cost of fatal opioid overdose, $ (millions)'] = df['Cost of fatal opioid overdose, $ (millions)'].replace({r'[^\d.]': ''}, regex=True)
df['Cost of fatal opioid overdose, $ (millions)'] = pd.to_numeric(df['Cost of fatal opioid overdose, $ (millions)'], errors='coerce')
df['Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)'] = df['Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)'].replace({r'[^\d.]': ''}, regex=True)
df['Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)'] = pd.to_numeric(df['Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)'], errors='coerce')
df['Per capita cost of opioid use disorder, $'] = df['Per capita cost of opioid use disorder, $'].replace({r'[^\d.]': ''}, regex=True)
df['Per capita cost of opioid use disorder, $'] = pd.to_numeric(df['Per capita cost of opioid use disorder, $'], errors='coerce')
df['Per capita cost of fatal opioid overdose, $'] = df['Per capita cost of fatal opioid overdose, $'].replace({r'[^\d.]': ''}, regex=True)
df['Per capita cost of fatal opioid overdose, $'] = pd.to_numeric(df['Per capita cost of fatal opioid overdose, $'], errors='coerce')
df['Per capita combined cost of opioid use disorder and fatal opioid overdose, $'] = df['Per capita combined cost of opioid use disorder and fatal opioid overdose, $'].replace({r'[^\d.]': ''}, regex=True)
df['Per capita combined cost of opioid use disorder and fatal opioid overdose, $'] = pd.to_numeric(df['Per capita combined cost of opioid use disorder and fatal opioid overdose, $'], errors='coerce')

# %% code cell 11
df.info()

# %% code cell 12
#central tendencies and varaince of combined between to to get idea of values
#mean, median, variance,
 #Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)
#Per capita combined cost of opioid use disorder and fatal opioid overdose, $
print("Combined cost of opioid use disorder and fatal opioid overdose, $ (millions) mean: ")
print(df.loc[:,"Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)"].mean())
print("Combined cost of opioid use disorder and fatal opioid overdose, $ (millions) median: ")
print(df["Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)"].median())
print("Combined cost of opioid use disorder and fatal opioid overdose, $ (millions) variance: ")
print(df["Combined cost of opioid use disorder and fatal opioid overdose, $ (millions)"].var())
print("Per capita combined cost of opioid use disorder and fatal opioid overdose mean: ")
print(df.loc[:,"Per capita combined cost of opioid use disorder and fatal opioid overdose, $"].mean())
print("Per capita combined cost of opioid use disorder and fatal opioid overdose median: ")
print(df["Per capita combined cost of opioid use disorder and fatal opioid overdose, $"].median())
print("Per capita combined cost of opioid use disorder and fatal opioid overdose variance: ")
print(df["Per capita combined cost of opioid use disorder and fatal opioid overdose, $"].var())

# %% code cell 13
df1 = pd.read_csv("OUD_and_FOO.csv")

# %% code cell 14
df1.head()

# %% code cell 15
df1.head()

# %% code cell 16
df1.describe()

# %% code cell 17
df1.info()

# %% code cell 18
print("FOO:Health care mean: ")
print(df1.loc[:,"FOO:Health care"].mean())
print("FOO:Health care median: ")
print(df1["FOO:Health care"].median())
print("FOO:Health care variance: ")
print(df1["FOO:Health care"].var())
print("FOO:Value of statistical life lost mean: ")
print(df1.loc[:,"FOO:Value of statistical life lost"].mean())
print("FOO:Value of statistical life lost median: ")
print(df1["FOO:Value of statistical life lost"].median())
print("FOO:Value of statistical life lost variance: ")
print(df1["FOO:Value of statistical life lost"].var())

# %% code cell 19
fig= plt.figure(figsize=(20, 15))
plt.barh(df1.loc[:,'Jurisdiction'],df1.loc[:,'Estimated case counts of opioid use disorder'])
plt.xlabel('Estimated case counts of opioid use disorder')
plt.ylabel('State')
plt.title('Distribution of Cases')

# %% [markdown] cell 20
# The above visualization is a pie chart about the proportion of each state does for the total amount of cases of Opiod Use Disorder. We can see that California, Texas, and Florida being the biggest contributors. This is interesting as the populations are different, but they all seem to contribute similar amounts to the total. Even with states with a ton of cities where people tend to associate drug use issues with lots of urban populations or blue states such as Illinois or California.

# %% code cell 21
fig = fig= plt.figure(figsize=(50, 15))
plt.scatter(x="Jurisdiction", y="FOO:Health care", s="FOO:Value of statistical life lost", data=df1)

# %% [markdown] cell 22
# This is a bubble plot that helps visualize values in a 2D type of scatter plot. This plot shows us the Jurisdiction, or the state, and the values of Fatal Opioid Overdoses cost in Health Care and Value of Statistical life lost. This graph demonstrates a trend that with more health care spent (higher on the y-axis) the state tends to have more of a higher value of life lost. This shows a great proportion between these and a correlation to these two variables in the spending and the importance of the poeple who did overdose due to opioid and was fatal.

# %% code cell 23
df1.head()

# %% code cell 24
df1['FOO sum'] = df1[['FOO:Health care', 'FOO:Lost productivity', 'FOO:Value of statistical life lost']].sum(axis=1)
df1['OUD sum'] = df1[['OUD:Health care', 'OUD:Substance use treatment', 'OUD:Criminal justice','OUD:Lost productivity','OUD:Reduced quality of life']].sum(axis=1)

# %% code cell 25
from sklearn.cluster import KMeans

df1_sub = df1[['OUD sum','FOO sum']]
kmeans = KMeans(n_clusters=6,random_state=42,init='k-means++',n_init=10)
kmeans.fit_transform(df1_sub)


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Centroids:", centroids)

# %% code cell 26
import matplotlib.pyplot as plt
plt.scatter(df1_sub['OUD sum'], df1_sub['FOO sum'], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x',s=200)
plt.xlabel("Opiod Use Disorder Cost Sum")
plt.ylabel("Fatal Opiod Overdose Cost Sum")
plt.title("K-means Clustering (k=6)")
plt.show()

# %% code cell 27
df1['labels'] = kmeans.labels_
df1

# %% code cell 28
print(enumerate(zip(df1['Jurisdiction'].values,kmeans.labels_)))

# %% code cell 29
from sklearn.metrics import silhouette_score
kmeans_labels = kmeans.fit_predict(df1_sub)
score = silhouette_score(df1_sub, kmeans_labels)

# %% code cell 30
score

# %% code cell 31
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df1.drop(['Estimated case counts of opioid use disorder','Jurisdiction'],axis=1)  # Replace with your feature columns
y = df1['Estimated case counts of opioid use disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# %% code cell 32
from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficients:', lr.coef_)

# %% code cell 33
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = df1['OUD:Criminal justice']
y = df1['OUD:Lost productivity']
z = df1['Estimated case counts of opioid use disorder']

ax.scatter(x, y, z, color='blue', label='Data Points', zorder=5)

A = np.vstack([x, y, np.ones_like(x)]).T
plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
x_plane, y_plane = np.meshgrid(np.linspace(x.min(), x.max(), 10),
                               np.linspace(y.min(), y.max(), 10))
z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane + plane_coef[2]

ax.plot_surface(x_plane, y_plane, z_plane, color='red', alpha=0.4, label='Regression Plane')

# Labels
ax.set_xlabel('OUD:Criminal justice')
ax.set_ylabel('OUD:Lost productivity')
ax.set_zlabel('Estimated case counts of opioid use disorder')
plt.title('Multiple Linear Regression')

plt.show()
