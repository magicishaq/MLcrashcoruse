#This model will create a binary classifcation to answer the question

#Are houses in this neighorhood above a certain price

%tensorflow_version 2.x

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt


pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


print('ran import statements')

#load datasets from the internet

#contains the training data
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

train_df = train_df.reindex(np.random.permutation(train_df.index)) #shuffles the training data

#Normalising values

#values will roughly be in the same range for multiple features
#to normalise the values, turn them into thier z-score ((value - mean) / standard deviation) includes the label

#calculate the z-scores of each column in the training set
train_df_mean = train_df.mean()
train_df_std = train_df.std() 
train_df_norm = (train_df - train_df_mean) / train_df_std

#look at the values

train_df_norm.head()

#do te same with the test data

test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std

test_df_norm.head()



#Create the binary label




