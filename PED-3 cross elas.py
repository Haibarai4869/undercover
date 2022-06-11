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

### chech example column 10001, use notna values
col = '10001'
df_x1 = df_x[df_x[col].notna()]
df_y1 = df_y[df_y[col].notna()]
#print(df_x1,df_y1)

### need to calculate coeff between column 10001 and the rest columns, need to find the rest first, remove billdate and 10001
col_list = df_x1.columns.tolist()
col_list.remove('billdate')
col_list.remove(col)
print(col_list)


eDict={
    "name":[],
    "elasticity":[]}


### find price and numsold in 10001 and 10002, check price 01 change then does num 02 change?
for column in col_list[:50]:
    sold_rec=[]
    for i in range(len(df_x1[column])):
        if not np.isnan(df_x1[column].iloc[i]) and not np.isnan(df_y1[column].iloc[i]):

            sold_rec.append([df_x1[column].iloc[i], df_y1[column].iloc[i]])
    df = pd.DataFrame(sold_rec, columns=['x_values', 'y_values'])
    #print(df)
### from the result above, check the groups x has changes y also differ, means x impacts y

    ### data is ready, fit it in model to calculate coeff, deal with missing values
    if not df.empty:
        x = sm.add_constant(df['x_values'])
        model = sm.OLS(df['y_values'], x)
        result = model.fit()

        try:
            intercept, slope = result.params
            if result.f_pvalue < 0.05:
                mean_price = np.mean(df['x_values'])
                mean_quan = np.mean(df['y_values'])
                eDict['name'].append(column)
                eDict['elasticity'].append(slope * (mean_price / mean_quan))
                print(f'cross elasticity for {col} / {column} - {slope * (mean_price / mean_quan)}')
                ### this print doesnt show results: p value too small or there have errors: use try...except
        except ValueError:
            print(f'{column} has a perfect elasticity or inelasticity')
            pass











