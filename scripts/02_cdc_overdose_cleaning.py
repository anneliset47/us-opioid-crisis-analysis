# Auto-exported from notebooks/02_cdc_overdose_cleaning.ipynb
# Source notebook retained for exploratory and narrative context.

# %% code cell 1
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% code cell 2
# Dataset from:
# https://www.cdc.gov/nchs/nvss/vsrr/prov-drug-involved-mortality.htm

# Load dataset
df = pd.read_csv("provisional_drug_overdose_death_counts_for_specific_drugs.csv", sep = ",")

# %% code cell 3
df.head(6)

# %% code cell 4
# Find duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

num_duplicates = duplicates.shape[0]
print(f"Number of duplicate rows: {num_duplicates}")

# %% code cell 5
# Check for missing values in each column
missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

# %% code cell 6
# Check data types
df.dtypes

# %% code cell 7
# Remove rows where the column jurisdiction_occurence is United States
index_df = df[df['jurisdiction_occurrence'] == 'United States'].index
df.drop(index_df, inplace=True)
df.reset_index(drop=True, inplace=True)
# df.head()

# %% code cell 8
# Remove unwanted columns
df = df.drop(['data_as_of', 'time_period', 'month_ending_date', 'footnote'], axis=1)
# df.head()

# %% code cell 9
# Rename columns
df = df.rename(columns={'death_year': 'Year',
                        'death_month': 'Month',
                        'jurisdiction_occurrence': 'Location',
                        'drug_involved': 'Drug',
                        'drug_overdose_deaths': 'Deaths'
                       })
# df.head()

# %% code cell 10
print(df.dtypes)

# %% code cell 11
min_year = df['Year'].min()
max_year = df['Year'].max()
print("Minimum Year:", min_year)
print("Maximum Year:", max_year)

# %% code cell 12
# Remove rows where the "Deaths" column is empty
df = df.dropna(subset=['Deaths'])

# df.head()

# %% code cell 13
# # Export CSV
# df.to_csv('drug_od_data', index=False)

# %% code cell 14
# Remove commas and convert 'Deaths' to integer data type
df['Deaths'] = df['Deaths'].str.replace(',', '').astype(int)
df.head(11)

# %% code cell 15
# Check data types
df.dtypes

# %% code cell 16
# Summary statistics of death counts by drug type

from scipy.stats import skew

# Compute summary statistics per drug type
summary_stats_per_drug = df.groupby("Drug")["Deaths"].agg(
    ["mean", "median", "var", "std", "min", "max", "count", "skew"]
)

# Rename columns
summary_stats_per_drug.rename(columns={
    "mean": "Mean",
    "median": "Median",
    "var": "Variance",
    "std": "Standard Deviation",
    "min": "Min",
    "max": "Max",
    "count": "Count",
    "skew": "Skewness",
}, inplace=True)

# Display the results
print(summary_stats_per_drug)

# %% code cell 17
# Boxplots of death counts by drug type

plt.figure(figsize=(10, 6))
df.boxplot(column='Deaths', by='Drug', vert=False, grid=False)
plt.title("Boxplot of Death Counts by Drug Type")
plt.suptitle("")
plt.xlabel("Deaths")
plt.ylabel("Drug Type")
plt.show()

# %% code cell 18
import scipy.stats as stats

# Get unique drug types
drugs = df["Drug"].unique()

# Create Q-Q plots for each drug type
fig, axes = plt.subplots(nrows=len(drugs), ncols=1, figsize=(8, 5 * len(drugs)))

for i, drug in enumerate(drugs):
    ax = axes[i] if len(drugs) > 1 else axes
    drug_data = df[df["Drug"] == drug]["Deaths"].dropna()

    stats.probplot(drug_data, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot for {drug}")

plt.tight_layout()
plt.show()

# %% code cell 19
df_grouped = df.groupby('Drug', as_index=False)['Deaths'].sum()

# Sort in descending order
df_grouped = df_grouped.sort_values(by='Deaths', ascending=False)

# Plot the aggregated deaths per drug
ax = sns.barplot(data=df_grouped, x='Deaths', y='Drug', orient='h')
plt.title("Total Overdose Deaths per Drug in The U.S. from 2019-2024")
plt.xlabel('Deaths in Millions')

# Add data labels at the end of each bar
for index, value in enumerate(df_grouped['Deaths']):
    ax.text(value, index, f'{value:,.0f}', va='center')

# Make room for data labels
plt.xlim(0, df_grouped['Deaths'].max() * 1.3)

plt.show()

# %% code cell 20
# List of opioid drugs
opioid_drugs = ["Fentanyl", "Heroin", "Oxycodone", "Morphine", "Hydrocodone", "Methadone", "Buprenorphine", "Tramadol"]

# Filter the dataset to only include opioid-related deaths
df_opioids = df[df['Drug'].isin(opioid_drugs)]

# Aggregate deaths per region and opioid drug
df_grouped_opioids = df_opioids.groupby(['Location', 'Drug'])['Deaths'].sum().unstack()

# Sort regions by total deaths (sum across opioid drugs)
df_grouped_opioids['Total_Deaths'] = df_grouped_opioids.sum(axis=1)
df_grouped_opioids = df_grouped_opioids.sort_values(by='Total_Deaths', ascending=True)
df_grouped_opioids = df_grouped_opioids.drop(columns=['Total_Deaths'])  # Remove extra column

df_grouped_opioids.plot(kind='barh', stacked=True, figsize=(10, 6))

plt.title("Total Opioid Overdose Deaths per Region in The U.S. (2019-2024)")
plt.xlabel("Total Deaths in Millions")
plt.ylabel("Region")
plt.legend(title="Opioid Drug", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()

# %% code cell 21
