# Auto-exported from notebooks/03_samhsa_demographics_cleaning.ipynb
# Source notebook retained for exploratory and narrative context.

# %% code cell 1
import pandas as pd
import seaborn as sns

df_treatment = pd.read_excel('reason_no_treatment.xlsx')

df_treatment = df_treatment.iloc[:, [0, 6]]
df_treatment = df_treatment.iloc[1:, :]

df_treatment.columns = ['reason', 'didnt_seek_treatment']

df_treatment.fillna(0)

# %% code cell 2
from wordcloud import WordCloud
import matplotlib.pyplot as plt

words = df_treatment['reason'].value_counts().to_dict()
word_cloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='coolwarm').generate_from_frequencies(words)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% code cell 3
heatmap_df = df_treatment.groupby('reason').value_counts().unstack()
sns.heatmap(heatmap_df, annot=True)

# %% [markdown] cell 4

# %% [markdown] cell 5

# %% code cell 6
df_sorted = df_treatment.sort_values(by="didnt_seek_treatment", ascending=False)
ax = sns.barplot(data = df_sorted, x = 'didnt_seek_treatment', y = 'reason', palette="mako")
plt.title("Reasons individuals did not seek mental health treatment")
ax.set_xlabel("Count")
ax.set_ylabel("")
plt.show()

# %% code cell 7
df_abuse = pd.read_excel('substance_or_mental_health_issue.xlsx')
df_abuse

df_abuse.columns = ['demographic', 'substance_problem22', 'substance_problem23', 'mental_problem22', 'mental_problem23']

df_abuse = df_abuse.iloc[11:18, :]

df_abuse.replace('*', 0, inplace=True)
df_abuse.replace('14.7a', 14.7, inplace=True)

df_abuse

pd.to_numeric(df_abuse['substance_problem22'])
pd.to_numeric(df_abuse['substance_problem23'])
pd.to_numeric(df_abuse['mental_problem22'])
pd.to_numeric(df_abuse['mental_problem23'])

df_abuse

#print(df_abuse.dtypes)

# %% code cell 8
summary_stats = df_abuse.describe()
print(summary_stats)

# %% code cell 9
plt.style.use('ggplot')
ax = df_abuse.boxplot(vert=False)
plt.title('Summary Statistics')
plt.tight_layout()
plt.show()

# %% code cell 10
df_abuse.plot(
    x="demographic", y=["substance_problem22", "substance_problem23"], kind="barh"
)
plt.title("Percentage of Substance Abuse Problems in 2022 vs 2023 by Demographic")
plt.xlabel("Percentage %")
plt.ylabel("")
legend = plt.legend()
legend.get_texts()[0].set_text('2022')
legend.get_texts()[1].set_text('2023')

# %% code cell 11
df_abuse.plot(
    x="demographic", y=["mental_problem22", "mental_problem23"], kind="barh"
)

plt.title("Percentage of Mental Health Problems in 2022 vs 2023 by Demographic")
plt.xlabel("Percentage %")
plt.ylabel("")
legend = plt.legend()
legend.get_texts()[0].set_text('2022')
legend.get_texts()[1].set_text('2023')
