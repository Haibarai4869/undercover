import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df_orderline = pd.read_csv('orderline.txt', delimiter = '\t', parse_dates = ['shipdate', 'billdate'])
df_orderline = df_orderline[['productid', 'billdate', 'unitprice', 'numunits']]
df_orderline = df_orderline.dropna()

df_l = df_orderline.groupby(['billdate', 'productid']).agg({'unitprice':np.mean,'numunits':np.sum})
df_l = df_l.reset_index()

x_pivot = df_l.pivot(index='billdate', columns='productid', values='unitprice')
df_x = pd.DataFrame(x_pivot.to_records())
y_pivot = df_l.pivot(index='billdate', columns='productid', values='numunits')
df_y = pd.DataFrame(y_pivot.to_records())

#print((df_x),(df_y))

### after loops done, save results here, ahead of loop:
eDict={
    "name":[],
    "elasticity":[]
}




### to find values in the two df: when there is a price, there is a sell
### in the two df, a spot in df 1 has a value, there is a value at same spot in df 2
### make a loop to take values out: 2 df have same structure, use one of them to make the loop
### culumn 1 is date, useless
### save the result in a list
### convert the new list to a df
for column in df_x.columns[1:]:
    sold_rec=[]
    for i in range(len(df_x[column])):
        if not np.isnan(df_x[column][i]) and not np.isnan(df_y[column][i]):
            sold_rec.append([df_x[column][i],df_y[column][i]])
    df=pd.DataFrame(sold_rec, columns= ['x_values', 'y_values'])

    x = sm.add_constant(df['x_values'])
    model = sm.OLS(df['y_values'],x)
    result = model.fit()

    try:
        intercept, slope = result.params

        if result.f_pvalue < 0.05:
            mean_price = np.mean(df['x_values'])
            mean_quan = np.mean(df['y_values'])

            eDict['name'].append(column)
            eDict['elasticity'].append(slope * (mean_price/mean_quan))
            print(f'elasticity for {column} - { slope * (mean_price/mean_quan)}')
    except ValueError:
        print(result.params)

df_2 = pd.DataFrame.from_dict(eDict)
#print(df_2)


### make this 3 records in order, add a ranking column, sort
df_2['ranking'] = df_2['elasticity'].rank(ascending=True).astype(int)
df_2.sort_values('elasticity', ascending=False, inplace=True)
#print(df_2)


### finaly, visualize it
plt.hlines(y=df_2['ranking'], xmin=0, xmax=[df_2['elasticity']], alpha = 0.5, linewidth=4)

for x, y, text in zip(df_2['elasticity'], df_2['ranking'], df_2['elasticity']):
    plt.text(x, y, round(text, 2))

plt.yticks(df_2['ranking'])
plt.show()
















