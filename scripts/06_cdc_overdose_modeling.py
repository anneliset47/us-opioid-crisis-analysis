# Auto-exported from notebooks/06_cdc_overdose_modeling.ipynb
# Source notebook retained for exploratory and narrative context.

# %% [markdown] cell 1
# # Import libraries

# %% code cell 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from sklearn.metrics import silhouette_score, davies_bouldin_score

# %% [markdown] cell 3
# # Import dataset

# %% code cell 4
df = pd.read_csv('drug_od_data')

# %% [markdown] cell 5
# # Model 1: K-Means Clustering

# %% code cell 6
# Step 1: Data Preparation
df['Deaths'] = df['Deaths'].astype(str).str.replace(',', '', regex=False).astype(float)
df_region = df[df['Location'] != 'United States']  # Exclude national total

# %% code cell 7
# Step 2: Pivot table (regions x drugs)
df_pivot = df_region.pivot_table(
    index='Location',
    columns='Drug',
    values='Deaths',
    aggfunc='sum',
    fill_value=0
)
display(df_pivot.head())

# %% code cell 8
# Step 3: K-Means Clustering
model_kmeans = KMeans(n_clusters=4, random_state=42)
df_pivot['Cluster'] = model_kmeans.fit_predict(df_pivot)

# %% code cell 9
# Step 4: View cluster assignment
display(df_pivot[['Cluster']].sort_values(by='Cluster'))

# %% code cell 10
# Step 5: Examine cluster profiles (avg deaths per drug per cluster)
df_profiles = df_pivot.groupby('Cluster').mean().round(2)
display(df_profiles)

# %% code cell 11
# Step 6: PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_pivot.drop(columns='Cluster'))

df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'], index=df_pivot.index)
df_pca['Cluster'] = df_pivot['Cluster']
df_pca['Region'] = df_pivot.index

# %% code cell 12
# Step 7: Plot clusters with region labels
plt.figure(figsize=(12, 8))

for cluster in sorted(df_pca['Cluster'].unique()):
    group = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(group['PCA1'], group['PCA2'], label=f'Cluster {cluster}')
    for _, row in group.iterrows():
        plt.text(row['PCA1'] + 0.2, row['PCA2'], row['Region'], fontsize=8)

plt.title('K-Means Clusters of Regions Based on Drug Overdose Patterns')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown] cell 13
# # Model 2: Apriori Frequent Pattern Mining

# %% code cell 14
# Step 1: Data Preparation
df['Deaths'] = df['Deaths'].astype(str).str.replace(',', '', regex=False).astype(float)
df_region = df[df['Location'] != 'United States']

# Group drugs by region and time (each basket = drugs reported in a region/month)
df_basket = df_region.groupby(['Location', 'Year', 'Month'])['Drug'] \
                     .apply(list).reset_index(name='Drug_List')
display(df_basket.head())

# %% code cell 15
# Step 2: Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(df_basket['Drug_List']).transform(df_basket['Drug_List'])
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

display(df_encoded.head())

# %% code cell 16
# Step 3: Apply Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

display(frequent_itemsets.head(10))

# %% code cell 17
# Step 4: Generate Association Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False)

# Create table
display(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# %% [markdown] cell 18
# # Performance Evaluation for K-Means Clustering

# %% code cell 19
# Silhouette Score and Davies-Bouldin Index
# Prepare data and labels
X = df_pivot.drop(columns='Cluster')
labels = df_pivot['Cluster']

# Compute metrics
sil_score = silhouette_score(X, labels)
db_index = davies_bouldin_score(X, labels)

# Show results
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_index:.3f}")

# %% [markdown] cell 20
# # Performance Evaluation for Apriori Frequent Pattern Mining

# %% code cell 21
# Support, confidence, and lift
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False)

# Create table
display(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
