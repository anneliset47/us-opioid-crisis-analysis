# Auto-exported from notebooks/08_per_capita_cost_modeling.ipynb
# Source notebook retained for exploratory and narrative context.

# %% code cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
pd.set_option('future.no_silent_downcasting', True)
from sklearn.inspection import DecisionBoundaryDisplay

# %% code cell 2
opioid_df = pd.read_excel('cost_opioid.xlsx')

# %% code cell 3
opioid_df

# %% code cell 4
# Dropping the combined columns, unnecessary
# Creating boolean target value for fatalities over 500
condition = opioid_df['case_fatal'] > 500
opioid_df['fatal_count500'] = condition

# %% code cell 5
opioid_df = opioid_df[['cost_fatal_opioid' , 'per_capita_fatal', 'fatal_count500']]
opioid_df = opioid_df.replace({True: 1, False: 0})
opioid_df['cost_fatal_opioid'] = opioid_df['cost_fatal_opioid'].astype(int)
opioid_df['fatal_count500'] = opioid_df['fatal_count500'].astype(bool)
opioid_df

# %% code cell 6
X = opioid_df.drop('fatal_count500', axis=1) 
y = opioid_df['fatal_count500']

# %% code cell 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

# %% code cell 8
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %% code cell 9
linear_accuracy = accuracy_score(y_test, y_pred)
linear_precision =  precision_score(y_test, y_pred)
linear_recall = recall_score(y_test, y_pred)
linear_f1 = f1_score(y_test, y_pred)

print(linear_accuracy, linear_precision, linear_recall, linear_f1)

# %% code cell 10
linear_matrix = confusion_matrix(y_test, y_pred)
matrix = ConfusionMatrixDisplay(confusion_matrix=linear_matrix)
matrix.plot()
plt.show()

# %% code cell 11
decision_function = clf.decision_function(X)

# Support Vectors

w = clf.coef_[0]
b = clf.intercept_[0]

x0 = np.linspace(5000, 10000)
decision_boundary = -w[0]/w[1] * x0 - b/w[1]

margin = 1/w[1]
high_gutter = decision_boundary + margin
low_gutter = decision_boundary - margin

# %% code cell 12
fig, ax = plt.subplots(figsize=(8, 4))
svs = clf.support_vectors_
plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
plt.plot(x0, decision_boundary, "k-", linewidth=1)
plt.plot(x0, high_gutter, "k--", linewidth=1)
plt.plot(x0, low_gutter, "k--", linewidth=1)
array_plot = opioid_df[['cost_fatal_opioid', 'per_capita_fatal']].to_numpy()
scatter = ax.scatter(array_plot[:, 0], array_plot[:, 1], s=150, c=y, label=y, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Fatalities count > 500")
ax.set_title("Cost of Fatal Opioid Overdose versus Per Capita Cost")

# %% code cell 13
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %% code cell 14
poly_accuracy = accuracy_score(y_test, y_pred)
poly_precision =  precision_score(y_test, y_pred)
poly_recall = recall_score(y_test, y_pred)
poly_f1 = f1_score(y_test, y_pred)

# %% code cell 15
print(poly_accuracy, poly_precision, poly_recall, poly_f1)

# %% code cell 16
poly_matrix = confusion_matrix(y_test, y_pred)
matrix = ConfusionMatrixDisplay(confusion_matrix=poly_matrix)
matrix.plot()
plt.show()

# %% code cell 17
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %% code cell 18
rbf_accuracy = accuracy_score(y_test, y_pred)
rbf_precision =  precision_score(y_test, y_pred)
rbf_recall = recall_score(y_test, y_pred)
rbf_f1 = f1_score(y_test, y_pred)

# %% code cell 19
print(rbf_accuracy, rbf_precision, rbf_recall, rbf_f1)

# %% code cell 20
fig, ax = plt.subplots(figsize=(8, 4))
array_plot = opioid_df[['cost_fatal_opioid', 'per_capita_fatal']].to_numpy()
scatter = ax.scatter(array_plot[:, 0], array_plot[:, 1], s=150, c=y, label=y, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Fatalities count > 500")
ax.set_title("Cost of Fatal Opioid Overdose versus Per Capita Cost")

# %% code cell 21
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# %% code cell 22
linked = linkage(X, 'ward')

# %% code cell 23
num_clusters = 3
threshold = max([linked[i, 2] for i in range(len(linked) - num_clusters + 1)])
plt.figure(figsize=(8, 5))
dendrogram(linked,
           color_threshold=0,
           above_threshold_color='gray',
           orientation='top')
plt.axhline(y=threshold, color='red', linestyle='--')
plt.title(f"3 Cluster Assignment")
plt.show()

# %% code cell 24
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(linked, num_clusters, criterion='maxclust')


plt.figure(figsize=(8, 4))
plt.scatter(array_plot[:, 0], array_plot[:, 1], c=clusters, cmap='Accent', s=80)
plt.title("Clustering for " f"k={num_clusters}")
plt.show()

# %% code cell 25
num_clusters = 4
threshold = max([linked[i, 2] for i in range(len(linked) - num_clusters + 1)])
plt.figure(figsize=(8, 5))
dendrogram(linked,
           color_threshold=0,
           above_threshold_color='gray',
           orientation='top')
plt.axhline(y=threshold, color='red', linestyle='--')
plt.title(f"4 Cluster Assignment")
plt.show()

# %% code cell 26
clusters = fcluster(linked, num_clusters, criterion='maxclust')


plt.figure(figsize=(8, 4))
plt.scatter(array_plot[:, 0], array_plot[:, 1], c=clusters, cmap='Accent', s=80)
plt.title("Clustering for " f"k={num_clusters}")
plt.show()

# %% code cell 27
num_clusters = 5
threshold = max([linked[i, 2] for i in range(len(linked) - num_clusters + 1)])
plt.figure(figsize=(8, 5))
dendrogram(linked,
           color_threshold=0,
           above_threshold_color='gray',
           orientation='top')
plt.axhline(y=threshold, color='red', linestyle='--')
plt.title(f"5 Cluster Assignment")
plt.show()

# %% code cell 28
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(linked, num_clusters, criterion='maxclust')


plt.figure(figsize=(8, 4))
plt.scatter(array_plot[:, 0], array_plot[:, 1], c=clusters, cmap='Accent', s=80)
plt.title("Clustering for " f"k={num_clusters}")
plt.show()

# %% code cell 29

# %% code cell 30
