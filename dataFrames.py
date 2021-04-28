#datafram is similar to an in-memory spreadsheet
#dataFrams stores data in cells
#Dataframe has named colums , numbered rows

import numpy as np
import pandas as pd

#creating a dataFrame

#create 5*2 dataFrame
my_data = np.array([[1, 2], [2, 2], [3, 4], [5, 6], [10, 20]])
my_col_names = ['temperature', 'activity']

#create the dataFrame
myDataFrame = pd.DataFrame(data=my_data, columns=my_col_names)

print(myDataFrame)

#adding a new column to the dataFrame

myDataFrame['new column'] = myDataFrame[
    'activity'] + 3  #creates a duplicate of activity but adds 3 to every entry

#print the first 3 rows
print(myDataFrame.head(3), '\n')

#just row 2
print(myDataFrame.iloc[[2]], '\n')

#rows 1 to 3
print(myDataFrame[1:4])

#jsut the activity column
print(myDataFrame['activity'])

print(myDataFrame)