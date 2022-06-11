import pandas as pd
import numpy as np

df = pd.read_csv('orders.txt', sep = '\t', parse_dates= ['orderdate'])


margin = 0.05
ac = 1

# total price / num of order = avg order price

customers = df.groupby('customerid').agg({"orderdate":lambda x: (x.max() - x.min()).days,
                                          "totalprice": lambda x: x.sum(),
                                          "orderid": lambda x: len(x) })

# this data, is not a transaction based business, original data shows 0 orderdates,
# one customer purchase once instead of frequent orders.
# went to last step, copied pasted customerid, made some same ids, looks like subscription customer orders
# so below, condition is make order date > 0, also able to calculate frequency

customers = customers[customers['orderdate']>0]
# print(customers)
'''
customerid  orderdate  totalprice  orderid                               
0                2538   312256.68     3424
45978             597      245.22        4
130971             85      107.25        4
'''

avg_order = customers['totalprice'].sum() / customers['orderid'].sum()
freq = customers['orderid'].sum() / customers['orderdate'].sum()

print(f' avg_order: {avg_order} and freq: {freq}')
'''
91.08658216783216 customerid
0          1.352246
45978      5.748744
130971    40.376471
DONT UNDERSTAND WHY FREQ FORMULA IS THAT
'''

# EVERY CUSTOMER HAS AN ID BUT THEY MAY NOT PURCHASE. GROUP CUSTOMER NEED TO GROUP BY HOUSEHOLD ID

# CHURN / retention
retention = customers[customers['orderid']>1].shape[0] / customers.shape[0]
print(retention)

##### final
# ltv = margin * avg_order * freq / churn - ac
print( margin * avg_order * freq / (1-retention) - ac)
























