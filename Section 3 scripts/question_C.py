import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
# Defining and prepping dataset
df = pd.read_csv("Section 2 Datasets/A.csv")
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

towns_affected = ['BUKIT PANJANG', 'BUKIT TIMAH']

# Limit dataset to 2000 - 2009
df = df[(df['month'] >= '2000') & (df['month'] <= '2009')]
df['treated'] = df['town'].apply(
    lambda x: 1 if x in towns_affected else 0)
le = LabelEncoder()
df['year_encoded'] = le.fit_transform(df['year'])
# Intervention period assumed to be 2008 July (announcement made)

# add 1 month for rounding up and reaction lag times
intervention_date = pd.to_datetime('2008-01')
df['post_intervention'] = df['month'].apply(
    lambda x: 1 if x >= intervention_date else 0)
df['did'] = df['treated'] * df['post_intervention']

# Correlation matrix to determine which features to include
corr_matrix = df[['resale_price', 'flat_type_encoded',
                  'floor_area_sqm', 'flat_model_encoded', 'year']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
# plt.show()

formula = "resale_price ~ treated + post_intervention + did  + floor_area_sqm + flat_model_encoded + year"

model = smf.ols(formula, data=df).fit()
print(model.summary())
