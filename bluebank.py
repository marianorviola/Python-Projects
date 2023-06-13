# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:29:28 2023

@author: maria
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Method 1 to read json data
json_file = open('loan_data_json.json')
data = json.load(json_file)

#Method 2 to read json data
with open('loan_data_json.json') as json_file:
    data = json.load(json_file)

#transform from list to dataframe
loandata = pd.DataFrame(data)

#Finding unique values in a column
loandata['purpose'].unique()

#describe the data in the dataset
loandata.describe()

#describe the data for a specific column
loandata['int.rate'].describe()
loandata['fico'].describe()

loandata['dti'].describe()

#using EXP() to get the annual income
income = np.exp(loandata['log.annual.inc'])

#Add converted income data as a new column to dataset
loandata['annualincome'] = income


#FICO Score
# fico >= 300 and < 400: 'Very Poor'
# fico >= 400 and ficoscore < 600: 'Poor'
# fico >= 601 and ficoscore < 660: 'Fair'
# fico >= 660 and ficoscore < 780: 'Good'
# fico >= 780: 'Excellent'

fico = 250

if fico >= 300 and fico < 400:
    ficocat = 'Very Poor'
elif fico >= 400 and fico < 600:
    ficocat = 'Poor'
elif fico >= 601 and fico < 660:
    ficocat = 'Fair'
elif fico >= 660 and fico < 700:
    ficocat = 'Good'
elif fico >= 700: 
    ficocat = 'Excellent'
else:
    ficocat = 'Unknown'
print(ficocat)
  
#for loops
#loops based on item or value

fruits = ['apple', 'pear', 'banana', 'cherry']

for x in fruits:
    print(x)
    y = x + ' fruit'
    print(y)

#loops based on position

for x in range(0,4):
    y = fruits[x]
    print(y)

for x in range(0,4):
    y = fruits[x] + ' for sale'
    print(y)


#apply for loops to loan data
#using first 10

length = len(loandata)
ficocat = []
for x in range(0, length):
    category = loandata['fico'][x]
    
    try:
        if category >= 300 and category < 400:
            cat = 'Very Poor'
        elif category >= 400 and category < 600:
            cat = 'Poor'
        elif category >= 601 and category < 660:
            cat = 'Fair'
        elif category >= 660 and category < 700:
            cat = 'Good'
        elif category >= 700:
            cat = 'Excellent'
        else:
            cat = 'Unknown'
    except:
        cat = 'Unknown'

    ficocat.append(cat)

#convert ficocat from a list to a series (for use as a column in the dataframe)
ficocat = pd.Series(ficocat)

#add ficocat as a column or a field in the loandata dataset
loandata['fico.category'] = ficocat


#df.loc as conditional statements
# df.loc[df[colmnname] condition, newcolumnname] = 'value if the condition is met'

# for interest rates, we create a new column, if rate > 0.12 then high, else low

loandata.loc[loandata['int.rate'] > 0.12, 'int.rate.type'] = 'High'
loandata.loc[loandata['int.rate'] <= 0.12, 'int.rate.type'] = 'Low'


#Bar plot showing Number of rows/loans by FICO category
catplot = loandata.groupby(['fico.category']).size()
catplot.plot.bar(color = 'green')
plt.show()

purposecount = loandata.groupby(['purpose']).size()
purposecount.plot.bar(color = 'red', width = 0.3)
plt.show()

#Scatter plot showing dti vs annual income

ypoint = loandata['annualincome']
xpoint = loandata['dti']
plt.scatter(xpoint, ypoint, color = 'green')
plt.show()


#writing to CSV file
loandata.to_csv('loan_cleaned.csv', index = True)










