import pandas as pd

df = pd.read_csv('candidate_1.tab.txt', delimiter = '\t')

'''
print(df.columns)
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
'''

df_input = df[['education', 'religion', 'research_area', 'professional',
       'pricing_group', 'race', 'age_group', 'gender']]

#print(df_input)

import seaborn as sns
import matplotlib.pyplot as plt

'''
sns.countplot(data = df_input, x = 'gender', hue = 'religion')
plt.show()
'''

print(pd.get_dummies(data= df_input, columns=df_input.columns))
#label encoding
#onehot encoding

import statsmodels.api as sm

olsModel = sm.OLS(df['selected'], pd.get_dummies(data= df_input, columns=df_input.columns))
res = olsModel.fit()
print(res.summary())





