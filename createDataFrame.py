import numpy as np
import pandas as pd


def gran():
    return np.random.randint(low=0, high=101)


#or

#myData = np.random.randint(low=0, high=100, size=(3,4))
myData = np.array([gran(), gran(), gran(), gran()],
                  [gran(), gran(), gran(), gran()],
                  [gran(), gran(), gran(), gran()])
colNames = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

myDataFrame = pd.DataFrame(data=myData, column=colNames)

print(myDataFrame)  #entire dataFrame
print(myDataFrame['Eleanor'][1], '\n')  #just row one

myDataFrane['Janet'] = myDataFrame['Tahani'] + myDataFrame['Jason']