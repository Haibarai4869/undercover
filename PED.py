import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df_orderline = pd.read_csv('orderline.txt', delimiter = '\t', parse_dates = ['shipdate', 'billdate'])
df_product = pd.read_csv('product.txt', delimiter = '\t')

'''
print(df_orderline.shape)
print('-----------------------------------------')
print(df_orderline.columns)
print('-----------------------------------------')
print(df_orderline.describe())
print('-----------------------------------------')
print(df_orderline.dtypes)
print('-----------------------------------------')
#print(df_orderline.isnull().count())

print(df_product.shape)
print('-----------------------------------------')
print(df_product.columns)
print('-----------------------------------------')
print(df_product.describe())
print('-----------------------------------------')
print(df_product.dtypes)
'''

df_orderline = df_orderline[['productid', 'billdate', 'unitprice', 'numunits']]
#print(df_orderline)

### want to know waht r unique groups of products, join two tables
df_pg = df_orderline.merge(df_product[['PRODUCTID', 'PRODUCTGROUPNAME']], left_on='productid', right_on='PRODUCTID')
df_pg = df_pg.drop(columns=['PRODUCTID'], axis=1)
#print(df_pg)

### in OLS linear regression, missing values can't be used, b4 lg, deal with or drop missing values
df_orderline = df_orderline.dropna()

### use OLS to make model
### ideal situation is low P value < 0.05, high R quare close to 0.9
### result has not bad P value, but very small R square, means products are being too broadly sold,/
### small coeff means products are mixed, so later should group it then check coeff
model = sm.OLS(df_orderline['numunits'], df_orderline['unitprice'])
results = model.fit()
#print(results.summary())

df_orderline_2 = df_orderline.groupby(['unitprice']).agg({'numunits':np.sum})
#print(df_orderline_2)

### in the above dataframe, unitprice is index. need to make index to a column
df_orderline_2 = df_orderline_2.reset_index()
#print(df_orderline_2)

### now check this grouped df linear reg info, fit it into the model
### in the result, coeff is bigger, good. P is too high! add a step of df adjustment ahead of building model:
### see df_orderline_2 df, there is 0 dollar products. it might be too old products. remove it.
df_orderline_2 = df_orderline_2[df_orderline_2['unitprice']>0]
#print(df_orderline_2)
model = sm.OLS(df_orderline_2['numunits'], df_orderline_2['unitprice'])
results = model.fit()
#print(results.summary())

'''
### want to see each product group info: make a loop of df, generate each sub-df
for x in df_pg['PRODUCTGROUPNAME'].unique():
    print(x)
    df = df_pg[df_pg['PRODUCTGROUPNAME'] == x]
    print(df)


### fit each sub-df to OLS model
for x in df_pg['PRODUCTGROUPNAME'].unique():
    print(f'-------------- {x} --------------')
    df = df_pg[df_pg['PRODUCTGROUPNAME'] == x]
    model = sm.OLS(df['numunits'], df['unitprice'])
    results = model.fit()
    print(results.summary())


### aggregate to optimize
for x in df_pg['PRODUCTGROUPNAME'].unique():
    print(f'-------------- {x} --------------')
    df = df_pg[df_pg['PRODUCTGROUPNAME'] == x]
    df = df.groupby(['unitprice']).agg({'numunits': np.sum})
    df = df.reset_index()
    df = df[df['unitprice'] > 0]
    model = sm.OLS(df['numunits'], df['unitprice'])
    results = model.fit()
    print(results.summary())
'''

### from above, OCCASION coeff is 2.65, highest. will be used.
### to see graph: plot_partregress_grid - statsmodels.formula.api
df = df_pg[df_pg['PRODUCTGROUPNAME'] == 'OCCASION']
df = df.groupby(['unitprice']).agg({'numunits': np.sum})
df = df.reset_index()
df = df[(df['unitprice'] > 0) & (df['numunits']<2000)]

model = smf.ols('numunits ~ unitprice', data = df)
results = model.fit()
print(results.summary())

fig = plt.figure(figsize=(12,8))
sm.graphics.plot_partregress_grid(results, fig= fig)
plt.show()
### graph interpret: need to remove outliers
### apply condition to numunits , <2000
### df = df[df['unitprice'] > 0 xxxxxxxxxxxxx].
### two conditions have priority, need to use () to equal both

### to see all info graphs, use sm.graphics.plot_regress_exog()
# sm.graphics.plot_regress_exog(results, 'unitprice', fig= fig)
# plt.show()

### all above is analysis on group of products.
### need to break it down, analize individual product.







