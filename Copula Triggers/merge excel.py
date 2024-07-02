# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:21:44 2023

@author: tomls
"""

import pandas as pd
import os

# create a list of Excel files to merge
excel_files = ['ML_cumulative_1_2015-01-01 to 2017-12-31.csv', 'ML_cumulative_1_2016-01-01 to 2019-1-1.csv', 'ML_cumulative_1_2017-01-01 to 2019-12-31.csv', 'ML_cumulative_1_2018-01-01 to 2020-12-30.csv','ML_cumulative_1_2019-01-01 to 2021-12-31.csv']

# create an empty DataFrame to hold the merged data
merged_data = pd.DataFrame()

# iterate over the Excel files and append their data to the merged_data DataFrame
for file in excel_files:
    # read the data from the Excel file into a DataFrame
    data = pd.read_csv(file)
    # append the data to the merged_data DataFrame
    merged_data = merged_data.append(data)

# save the merged data to a new Excel file
merged_data.to_excel('merged_data_1.xlsx', index=False)

# print a message to confirm that the merging was successful
print('Excel files merged successfully!')
