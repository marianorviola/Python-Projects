# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:23:56 2023

@author: maria
"""

import pandas as pd

# file_name = pd.read_csv('file.csv') <-- format of read_csv

data = pd.read_csv('transaction2.csv')

data = pd.read_csv('transaction2.csv', sep=';')

# Summary of the data
data.info()

#Working with calculations
#Defining the variables

CostPerItem = 11.73
SellingPricePerItem = 21.11
NumberOfItemsPurchased = 6

#Mathematical Operations on Tableau
ProfitPerItem = 21.11 - 11.73
ProfitPerItem = SellingPricePerItem - CostPerItem

ProfitPerTransaction = ProfitPerItem * NumberOfItemsPurchased
CostPerTransaction = CostPerItem * NumberOfItemsPurchased
SalesPerTransaction = SellingPricePerItem * NumberOfItemsPurchased

#CostPerTransaction Column Calculation
#CostPerTransaction = CostPerItem * NumberOfItemsPurchased
# variable = dataframe['column_name']

CostPerItem = data['CostPerItem']
NumberOfItemsPurchased = data['NumberOfItemsPurchased']
CostPerTransaction = CostPerItem * NumberOfItemsPurchased

#Adding a new column to the data frame
data['CostPerTransaction'] = CostPerTransaction
data['CostPerTransaction'] = data['CostPerItem'] * data['NumberOfItemsPurchased']

#SalesPerTransaction
data['SalesPerTransaction'] = data['SellingPricePerItem'] * data['NumberOfItemsPurchased']

#ProfitPerTransaction
data['ProfitPerTransaction'] = data['SalesPerTransaction'] - data['CostPerTransaction']

#Markup = (Sales - Cost) / Cost
data['Markup'] = data['ProfitPerTransaction'] / data['CostPerTransaction']

#Rounding Markup
#roundmarkup = round(data['Markup'], 2)

#Add the rounded Markup as a new column in the data frame
data['Markup'] = round(data['Markup'],2)

#Combining data fields
my_name = 'Dee' + ' Naidoo'
my_date = 'Day'+'-'+'Month'+'-'+'Year'

#Checking column data type
print(data['Day'].dtype)

#Change 'Day' column from int to string
day = data['Day'].astype(str)

print(day.dtype)

my_date = day +'-'+ data['Month']

#Check 'Year' column data type
print(data['Year'].dtype)

#Change 'Year' column from int to string
year = data['Year'].astype(str)

print(year.dtype)

my_date = day +'-'+ data['Month'] +'-'+year

data['date'] = my_date

#Using iloc to view specific columns/rows

data.iloc[0] #views the row with index = 0
data.iloc[0:3] #views the first 3 rows
data.iloc[-5:] #views the last 5 rows

data.head(5) #views the first 5 rows
data.iloc[:,2] #views all rows in column 2
data.iloc[4,2] #views row 4 in column 2

#Use split to split the 'ClientKeywords' field or column
#new_var = column.str.split('sep', expand = True)

split_col = data['ClientKeywords'].str.split(',' , expand = True)

#create new columns for split columns in 'ClientKeywords' column
data['ClientAge'] = split_col[0]
data['ClientType'] = split_col[1]
data['LengthOfContract'] = split_col[2]

#Use replace function
data['ClientAge'] = data['ClientAge'].str.replace('[','')
data['LengthOfContract'] = data['LengthOfContract'].str.replace(']','')

#Use the lower function to change item to lowercase
data['ItemDescription'] = data['ItemDescription'].str.lower()

#How to merge files
#Bringing in a new dataset

seasons = pd.read_csv('value_inc_seasons.csv', sep=';')

#Merging files: merge_df = pd.merge(df_old, df_new, on = 'key')

data = pd.merge(data, seasons, on = 'Month')

#Dropping columns
# df = df.drop('columnname', axis = 1)

data = data.drop('ClientKeywords', axis = 1)
data = data.drop('Day', axis = 1)

data = data.drop(['Year', 'Month'], axis = 1)

#Export into a CSV file

data.to_csv('ValueInc_Cleaned.csv', index = False)





























