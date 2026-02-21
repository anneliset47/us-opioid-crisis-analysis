# Auto-exported from notebooks/05_cms_prescribing_modeling.ipynb
# Source notebook retained for exploratory and narrative context.

# %% [markdown] cell 1
# ### Dataset 1: Analyzing Opioid Prescription Rates by State and Provider Type
#
# Implementing K-Means clustering and naive bayes classification to explore provider presecription patterns by state and specialty.
#
# #### Data Cleaning

# %% code cell 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder

# %% code cell 3
## load data and examine head of data
import pandas as pd
filename = "prescription_providers.csv"
df = pd.read_csv(filename)
df.head()

# %% code cell 4
df.shape

# %% code cell 5
print(df.columns)

# %% code cell 6
df = df[['Prscrbr_Crdntls','Prscrbr_State_FIPS','Prscrbr_RUCA','Prscrbr_Type','Opioid_Tot_Clms', 'Opioid_Tot_Drug_Cst',
        'Opioid_Tot_Suply','Opioid_Prscrbr_Rate','Opioid_LA_Tot_Clms','Opioid_LA_Prscrbr_Rate']]
df.head()

# %% code cell 7
df.isna().sum()

# %% code cell 8
# too many NA values
df = df.drop(columns='Opioid_LA_Prscrbr_Rate')

# %% code cell 9
df = df.dropna(subset=['Opioid_Prscrbr_Rate'])

# %% code cell 10
# drop all rows with NA values, dataset is large and this is a small percentage
df = df.dropna()

# %% code cell 11
df.shape

# %% code cell 12
df.head(10)

# %% [markdown] cell 13
# ### K-Means Clustering

# %% code cell 14
# for clustering, drop categorical columns
df_clustering = df.drop(columns = ['Prscrbr_Crdntls','Prscrbr_State_FIPS','Prscrbr_RUCA','Prscrbr_Type'])
df_clustering.head()

# %% code cell 15
# consolidate to 3 features for visualization
df_clustering = df_clustering.drop(columns = ['Opioid_LA_Tot_Clms','Opioid_Tot_Suply'])
df_clustering.head()

# %% code cell 16
# normalize data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # standardizes to zero mean and unit variance
df_clustering = pd.DataFrame(scaler.fit_transform(df_clustering), columns=df_clustering.columns)
df_clustering.head()

# %% code cell 17
# filter out outliers outside of IQR
Q1 = df_clustering.quantile(0.25)
Q3 = df_clustering.quantile(0.75)
IQR = Q3 - Q1
df_clustering = df_clustering[~((df_clustering < (Q1 - 1.5 * IQR)) | (df_clustering > (Q3 + 1.5 * IQR))).any(axis=1)]

# %% code cell 18
df_clustering = df_clustering.sample(frac=0.1, random_state=42)

# %% code cell 19
import matplotlib.pyplot as plt

# %% code cell 20
# calculate silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
k_values = range(2, 10)  # k from 2 to 9

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_clustering)
    silhouette_scores.append(silhouette_score(df_clustering, labels))

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.show()

# %% code cell 21
inertias = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_clustering)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

# %% code cell 22
# fit k means on scaled 3d data
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(df_clustering)

# Get cluster centers
centroids = kmeans.cluster_centers_

 # Extract the 3 columns for visualization
x, y, z = df_clustering.iloc[:, 0], df_clustering.iloc[:, 1], df_clustering.iloc[:, 2]

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Plot data points
scatter = ax.scatter(x, y, z, c=labels, cmap="viridis", edgecolor='k', s=20, alpha=0.5)

# Plot centroids (black X markers)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
               c='red', s=300, marker='X', edgecolors='black', label='Centroids')

    # Labels and Title
ax.set_xlabel('Total Opioid Claims')
ax.set_ylabel('Total Opioid Cost')
ax.set_zlabel('Opioid Prescription Rate')
ax.set_title(f'K-Means Clustering (k=3)')

# Show color bar for cluster labels
plt.colorbar(scatter, label="Cluster Label")

# Show the plot
plt.legend()
plt.show()

# %% code cell 23
df_clustering.shape

# %% [markdown] cell 24
# #### Decision Tree

# %% code cell 25
df['Prscrbr_Crdntls'].value_counts()

# %% code cell 26
type_counts = df['Prscrbr_Crdntls'].value_counts()
print(type_counts.to_string()) 

# %% code cell 27
#drop data with specialties of less than 50 data points

credential_counts = df['Prscrbr_Crdntls'].value_counts()

# filter for specialties with 50 or more data points
valid_specialties = credential_counts[credential_counts >= 99].index

# keep only rows with valid specialties
df_dt = df[df['Prscrbr_Crdntls'].isin(valid_specialties)]

# %% code cell 28
type_counts = df_dt['Prscrbr_Crdntls'].value_counts()
print(type_counts.to_string()) 

# %% code cell 29
# a lot of variation in format of reporting
#need to remap for consistency
# dictionary to map raw credentials to standardized labels
credential_mapping = {
    # MD
    'MD': 'MD', 'M.D.': 'MD', 'M.D': 'MD', 'M. D.': 'MD', 'MD, PHD': 'MD', 'MD PHD': 'MD',
    'MD, MPH': 'MD', 'M.D., PH.D.': 'MD', 'M.D., M.P.H.': 'MD', 'MD, MS': 'MD',
    'M.D.,': 'MD', 'M.D., PHD': 'MD', 'M.D., MPH': 'MD', 'MD.': 'MD', 'M.D,': 'MD',
    'M.D., M.S.': 'MD', 'MD MPH': 'MD', 'MD/PHD': 'MD',

    # DO
    'DO': 'DO', 'D.O.': 'DO', 'D.O': 'DO', 'DO, MPH': 'DO',

    # PA
    'PA-C': 'PA-C', 'PA': 'PA-C', 'P.A.': 'PA-C', 'PAC': 'PA-C', 'P.A': 'PA-C',
    'P.A.-C': 'PA-C', 'P.A.-C.': 'PA-C', 'PA C': 'PA-C', 'P.A-C': 'PA-C',
    'RPAC': 'PA-C', 'MPAS, PA-C': 'PA-C', 'PA-C, MPAS': 'PA-C', 'PA-C, ATC': 'PA-C',
    'MMS, PA-C': 'PA-C', 'MS, PA-C': 'PA-C', 'PHYSICIAN ASSISTANT': 'PA-C',

    # NP 
    'NP': 'NP', 'N.P.': 'NP', 'NP-C': 'NP', 'N.P': 'NP', 'NP-BC': 'NP',
    'NURSE PRACTITIONER': 'NP', 'RN, NP': 'NP', 'APRN, NP-C': 'NP', 'MSN, NP-C': 'NP',

    # FNP
    'FNP': 'FNP', 'FNP-C': 'FNP', 'FNP-BC': 'FNP', 'F.N.P.': 'FNP', 'CFNP': 'FNP',
    'FNPC': 'FNP', 'RN, FNP': 'FNP', 'RN, FNP-C': 'FNP', 'MSN, FNP-C': 'FNP',
    'MSN, FNP-BC': 'FNP', 'MSN, FNP': 'FNP',

    # APRN
    'APRN': 'APRN', 'APN': 'APRN', 'APRN, FNP-C': 'APRN', 'APRN-CNP': 'APRN',
    'APNP': 'APRN', 'APRN, CNP': 'APRN', 'APRN, FNP-BC': 'APRN', 'APRN, PMHNP-BC': 'APRN',
    'APRN-BC': 'APRN', 'APRN-C': 'APRN', 'APRN FNP-C': 'APRN', 'A.P.R.N.': 'APRN',
    'APRN, BC': 'APRN', 'APRN, FNP': 'APRN', 'APRN-NP': 'APRN',

    # CRNP
    'CRNP': 'CRNP', 'C.R.N.P.': 'CRNP', 'CRNP-PMH': 'CRNP',

    # ANP
    'ANP': 'ANP', 'ANP-BC': 'ANP', 'ANP-C': 'ANP',

    # PMHNP
    'PMHNP': 'PMHNP', 'PMHNP-BC': 'PMHNP', 'DNP, PMHNP-BC': 'PMHNP',

    # AGNP
    'AGNP': 'AGNP', 'AGPCNP': 'AGNP', 'AGACNP-BC': 'AGNP', 'AGNP-C': 'AGNP',
    'AGPCNP-BC': 'AGNP', 'AGACNP': 'AGNP', 'ACNPC-AG': 'AGNP',

    # CNP
    'CNP': 'CNP', 'C.N.P.': 'CNP', 'RN, CNP': 'CNP',

    # CNM
    'CNM': 'CNM', 'C.N.M.': 'CNM',

    # WHNP
    'WHNP': 'WHNP', 'WHNP-BC': 'WHNP',

    # CRNA
    'CRNA': 'CRNA',

    # DNP
    'DNP': 'DNP', 'DNP, FNP-C': 'DNP', 'DNP, FNP-BC': 'DNP', 'DNP, APRN, FNP-C': 'DNP',
    'DNP, APRN, FNP-BC': 'DNP', 'DNP, APRN': 'DNP', 'DNP, APRN, PMHNP-BC': 'DNP',

    # DDS
    'DDS': 'DDS', 'D.D.S.': 'DDS', 'D.D.S': 'DDS', 'DDS, MS': 'DDS', 'DDS MS': 'DDS',
    'DDS, MD': 'DDS', 'D.D.S., M.S.': 'DDS', 'D.D.S., M.D.': 'DDS',

    # DMD
    'DMD': 'DMD', 'D.M.D.': 'DMD', 'D.M.D': 'DMD', 'DMD, MD': 'DMD', 'DMD, MS': 'DMD',

    # OD
    'OD': 'OD', 'O.D.': 'OD', 'O.D': 'OD', 'O. D.': 'OD',

    # DPM
    'DPM': 'DPM', 'D.P.M.': 'DPM', 'D.P.M': 'DPM',

    # PharmD
    'PHARMD': 'PharmD', 'PHARM.D.': 'PharmD', 'PHARM.D': 'PharmD', 'PHARM D': 'PharmD',
    'PHARM. D.': 'PharmD', 'PHARM D.': 'PharmD', 'PHARM. D': 'PharmD', 'PHARMD.': 'PharmD',
    'PHARMD, RPH': 'PharmD', 'PHARMACIST': 'PharmD',

    # RPH
    'RPH': 'RPh', 'R.PH.': 'RPh', 'R.PH': 'RPh',

    # RN
    'RN': 'RN',

    # PHD
    'PHD': 'PhD', 'M.D., PH.D': 'PhD',

    # MBBS
    'MBBS': 'MBBS', 'M.B.B.S.': 'MBBS', 'M.B.B.S': 'MBBS',

    # ND
    'ND': 'ND', 'N.D.': 'ND', 'NMD': 'ND',
}

# %% code cell 30
df_dt['Prscrbr_Crdntls_Cleaned'] = df_dt['Prscrbr_Crdntls'].map(credential_mapping).fillna(df_dt['Prscrbr_Crdntls'])

# %% code cell 31
type_counts = df_dt['Prscrbr_Crdntls_Cleaned'].value_counts()
print(type_counts.to_string()) 

# %% code cell 32
# reduce dataset to providers with at least 7500 data points

providers_to_keep = [
    'MD', 'PA-C', 'DO', 'DDS', 'NP', 'FNP',
    'OD', 'APRN', 'DMD', 'PharmD', 'DPM',
    'ARNP', 'CRNP'
]

df_dt = df_dt[df_dt['Prscrbr_Crdntls_Cleaned'].isin(providers_to_keep)]

# %% code cell 33
# balance data, using both under and oversampling so that each provider type has 20,000 data points
from sklearn.utils import resample

balanced = []
for provider in providers_to_keep:
    subset = df_dt[df_dt['Prscrbr_Crdntls_Cleaned'] == provider]
    if len(subset) > 20000:
        # undersample
        sampled = resample(subset,
                           replace=False,
                           n_samples=20000,
                           random_state=42)
    else:
        # oversample
        sampled = resample(subset,
                           replace=True,
                           n_samples=20000,
                           random_state=42)
    balanced.append(sampled)

# combine into one balanced dataset
df_dt_2 = pd.concat(balanced)

# %% code cell 34
# check work
type_counts = df_dt_2['Prscrbr_Crdntls_Cleaned'].value_counts()
print(type_counts)

# %% code cell 35
df_dt_2.head()

# %% code cell 36
df_dt_2=df_dt_2.drop(columns= 'Prscrbr_Crdntls', axis = 1)

# %% code cell 37
df_dt_2.head()

# %% code cell 38
specialty_counts = df_dt_2['Prscrbr_Type'].value_counts()
print(specialty_counts.to_string()) 

df_dt_2.drop(columns = 'Prscrbr_Type', axis = 1)

# %% code cell 39

# filter for specialties with 50 or more data points
valid_specs = specialty_counts[specialty_counts >= 99].index

# keep only rows with valid specialties
df_dt_2 = df_dt_2[df_dt_2['Prscrbr_Type'].isin(valid_specs)]

# %% code cell 40
# create training and testing data, remove and save label
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# %% code cell 41
#encode categorical variables
from sklearn.preprocessing import LabelEncoder

# initialize encoders
le_type = LabelEncoder()
le_cred = LabelEncoder()

# fit and transform the columns
df_dt_2['Prscrbr_Type_Encoded'] = le_type.fit_transform(df_dt_2['Prscrbr_Type'])
df_dt_2['Prscrbr_Crdntls_Encoded'] = le_cred.fit_transform(df_dt_2['Prscrbr_Crdntls_Cleaned'])

type_mapping = dict(zip(le_cred.transform(le_cred.classes_), le_cred.classes_))
print("Prscrbr_Type Mapping:")
print(type_mapping)

#drop categorical columns
df_dt_2 = df_dt_2.drop(['Prscrbr_Type', 'Prscrbr_Crdntls_Cleaned'], axis=1)

# %% code cell 42
df_dt_2.head(10)

# %% code cell 43
training, testing = train_test_split(df_dt_2, test_size=.3)
##  Save the Labels and then remove them from the Training and Testing data
training_label = training["Prscrbr_Crdntls_Encoded"]
training=training.drop(["Prscrbr_Crdntls_Encoded"], axis=1)
testing_label = testing["Prscrbr_Crdntls_Encoded"]
testing=testing.drop(["Prscrbr_Crdntls_Encoded"], axis=1)

# %% code cell 44
# initialize and train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42,max_depth=15,min_samples_split=10)
clf.fit(training, training_label)  # Training the model with training data

# Step 3: Make predictions on the test set
prediction = clf.predict(testing)  # Predict labels for the test data

# Step 4: Evaluate the model
accuracy = accuracy_score(testing_label, prediction)
print(f"Accuracy: {accuracy:.2f}")

# Print additional evaluation metrics
print("\nClassification Report:")
print(classification_report(testing_label, prediction))

dt_cf = confusion_matrix(testing_label, prediction)
print("\nConfusion Matrix:")
print(confusion_matrix(testing_label, prediction))

# %% code cell 45
## Create and display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cf,display_labels=clf.classes_)
disp.plot()
plt.title("Provider Type Decision Tree Confusion Matrix")
plt.show()

# %% code cell 46
