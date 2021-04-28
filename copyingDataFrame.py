import numpy as np
import pandas as pd

#copying a dataFrame
#referencing
#copying by calling pd.DataFrame.copy you create a true independant copy

myData = np.random.randint(low=10, high=100, size=(10, 4))
colNames = list('abcd')

df = pd.DataFrame(data=myData, columns=colNames)

print(df)

reference = df

trueCopy = df.copy()
