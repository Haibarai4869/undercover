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

#print(pd.get_dummies(data= df_input, columns=df_input.columns))
#label encoding
#onehot encoding

import statsmodels.api as sm

olsModel = sm.OLS(df['selected'], pd.get_dummies(data= df_input, columns=df_input.columns))
res = olsModel.fit()


#print(res.params.items)
#print(res.pvalues)

df_res = pd.DataFrame(
       {
              'name':res.params.keys(),
              'coeff':res.params.values,
              'pvalue':res.pvalues
       })

#print(df_res)

'''
import numpy as np

plt.title('part worth')
pwu = df_res['pvalue']
xbar = np.arange(len(pwu))
plt.barh(xbar, df_res['pvalue'])
plt.show()
'''

'''
for x in res.params.items():
       print(x)

a = 33333
print(a)

for key, coeff in res.params.items():
       print(key.split('_'))
'''

feature_range = dict()
for key, coeff in res.params.items():
       feature = key.split('_')[0]
       if feature not in feature_range:
              feature_range[feature] = list()

       feature_range[feature].append(coeff)
print(feature_range)


feature_importance =\
       { key: round(max(value)-min(value), 2) for key, value in feature_range.items()}
print(feature_importance)


total_importance = sum(feature_importance.values())

feature_relative_importance = {
key: round(value/total_importance, 2) * 100 for key, value in feature_importance.items()
}
print(feature_relative_importance)


# sns needs a dataframe, need to change dic to df
df_feature_imp = pd.DataFrame(list(feature_importance.items()),
                              columns=['feature', 'importance'])\
       .sort_values(by='importance', ascending= False)

df_feature_relativeimp = pd.DataFrame(list(feature_relative_importance.items()),
                              columns=['feature', 'importance'])\
       .sort_values(by='importance', ascending= False)


#sns.barplot(data = df_feature_imp, x='feature', y='importance')
sns.barplot(data = df_feature_relativeimp, x='feature', y='importance')
plt.show()


















