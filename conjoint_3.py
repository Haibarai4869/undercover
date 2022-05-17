import pandas as pd
import numpy as np

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

pd.options.display.float_format = '{:.2f}'.format
#print(df_res)

df_res['abs_coeff'] = np.abs(df_res['coeff'])
df_res['sig_95'] = df_res['pvalue']< 0.05
df_res['color'] = ['b' if x else 'r' for x in df_res['sig_95']]
df_res = df_res.sort_values(by='abs_coeff', ascending=True)
df_att_level = pd.get_dummies(data=df_input, columns=df_input.columns)
#print(df_att_level)

### to find out Utility, needs to use dummy to make attributes and levels combination
### use OLS to find out Coeff
### sum of (attributes / levels) for each record = each Utility
for col in df_att_level:
    df_att_level.loc[:, col] = df_att_level.loc[:, col] * df_res.loc[col, 'coeff']

#print(df_att_level)

### Let's calculate Utility score
utility_score = df_att_level.sum(axis = 1)

prefered_index = np.argmax(utility_score)
print(df_att_level.loc[prefered_index])
print(df.loc[prefered_index])


total_uscore = 0
rational = 0.9
for x in utility_score:
    total_uscore = total_uscore + np.exp(x * rational)

pos = 0
for x in utility_score:
    prob = np.exp(x * rational)/total_uscore
    print(f'market share for {pos} is {prob*100} %')
    pos = pos + 1


   
      

### to do plot is to find out which attributes are important
'''
f, ax = plt.subplots(figsize = (14, 8))
plt.title('part worth')
pwu = df_res['pvalue']
xbar = np.arange(len(pwu))

plt.barh(xbar, df_res['coeff'], color=df_res['color'])
plt.ysticks(xbar, labels=df_res['name'])

plt.show()



for x in res.params.items():
       print(x)

a = 33333
print(a)

for key, coeff in res.params.items():
       print(key.split('_'))





feature_range = dict()
for key, coeff in res.params.items():
       feature = key.split('_')[0]
       if feature not in feature_range:
              feature_range[feature] = list()

       feature_range[feature].append(coeff)
print(feature_range)


feature_importance =\
       { key: round(max(value)-min(value), 2) for key, value in feature_range.items()}
#print(feature_importance)


total_importance = sum(feature_importance.values())
feature_relative_importance = {
key: round(value/total_importance, 2) * 100 for key, value in feature_importance.items()}
#print(feature_relative_importance)


### sns needs a dataframe, need to change dic to df
df_feature_imp = pd.DataFrame(list(feature_importance.items()),
                              columns=['feature', 'importance'])\
       .sort_values(by='importance', ascending= False)

df_feature_relativeimp = pd.DataFrame(list(feature_relative_importance.items()),
                              columns=['feature', 'importance'])\
       .sort_values(by='importance', ascending= False)

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (14, 8))
sns.barplot(data = df_feature_imp, x='feature', y='importance', ax = ax1)
sns.barplot(data = df_feature_relativeimp, x='feature', y='importance', ax = ax2)
plt.show()
'''



