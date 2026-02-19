## U.S. Opioid Crisis: Machine Learning Analysis of Prescribing, Overdose Trends, and Economic Impact

### Overview:
This project applies statistical analysis and machine learning to investigate prescribing behavior, overdose mortality patterns, and the economic burden of opioid use disorder (OUD) in the United States.

Using multi-source federal public health datasets, we identify structural drivers of overdose risk and state-level variation to inform public health strategy and policy allocation.

### Project Website
For a full walkthrough of the analysis, visualizations, and narrative findings, visit:
🔗 [Exploring the Opioid Crisis](https://sites.google.com/view/data-mining-project-group-one/introduction)

### Key Questions:
- How are opioid prescription patterns changing over time?
- Do prescribing behaviors vary by provider credential or geography?
- What regional overdose patterns exist by drug type?
- Which drugs most frequently co-occur in fatal overdoses?
- How do healthcare and criminal justice costs relate to overdose burden?
- Can states be classified by overdose severity using economic indicators?

### Tools & Methods: 
#### Languages & Libraries:
- Python
- pandas, NumPy
- scikit-learn
- matplotlib / seaborn
- BeautifulSoup (web scraping)
#### Machine Learning Models:
- K-Means Clustering
- Decision Tree Classification
- Multiple Linear Regression
- Support Vector Machines (SVM)
- Hierarchical Clustering
- Apriori Frequent Pattern Mining
- PCA for dimensionality reduction

### Core Findings: 
#### 1. Prescription Trends
- Overall opioid prescription rates have steadily declined since 2013.
- The proportion of long-acting opioids increased significantly after 2017 — a notable risk factor for overdose.
#### 2. Provider Behavior
- Prescribing patterns vary by provider credential.
- A decision tree classified provider type with 76% accuracy based on prescribing behavior and geography.
- Behavioral clustering revealed high-cost, high-rate prescriber groups.
#### 3. Overdose Patterns
- Fentanyl is the dominant contributor to overdose mortality across all regions.
- Strong co-occurrence exists between fentanyl, heroin, and methamphetamine.
- Regional clustering revealed distinct geographic drug profiles.
#### 4. Economic Impact
- Criminal justice and productivity costs are strong predictors of OUD case burden.
- States cluster into distinct spending profiles for OUD and fatal overdose.
- SVM classification achieved up to 100% accuracy distinguishing high-fatality states using per capita cost features.

### Technical Highlights: 
- Outlier handling using IQR filtering
- Feature scaling and encoding for model integrity
- Hyperparameter tuning (Elbow method, silhouette score optimization, kernel testing)
- Model evaluation via RMSE, accuracy, silhouette score, and confusion matrices
- Dimensionality reduction via PCA for high-dimensional clustering

### Data Sources: 
- Centers for Medicare & Medicaid Services (CMS)
- CDC Provisional Drug Overdose Mortality Data
- CDC Cost of Opioid Use Disorder Reports
- National Survey on Drug Use and Health (SAMHSA)

### Limitations: 
- Aggregated state-level data may introduce ecological bias.
- HIPAA constraints limit patient-level granularity.
- Models should inform, not replace, public health expertise.

### Impact: 
This project demonstrates the application of supervised and unsupervised machine learning to large-scale public health datasets, translating complex opioid crisis data into actionable, policy-relevant insights.
