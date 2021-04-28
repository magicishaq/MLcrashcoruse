# Learning objectives
# After doing this Colab, you'll know how to do the following:

# Split a training set into a smaller training set and a validation set.
# Analyze deltas between training set and validation set results.
# Test the trained model with a test set to determine whether your trained model is overfitting.
# Detect and fix a common training problem.

# using the right version of tensorFlow
#@title Run on TensorFlow 2.x
%tensorflow_version 2.x

import numpy as np
import pandas as pd
import tensorflow as tf
from matplot import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#load the datasets from the internet
#contains the training set data
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
#contains the test set data
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")



#scaling the labels
scale_factor = 1000.0
#scale the training sets label
train_df["median_house_value"] /= scale_factor
test_df["median_house_value"] /= scale_factor



#load functions that build and train a model
#defines the models topography
def build_model(my_learning_rate):
    #most simple model
    model=tf.keras.Sequential()
    #adds one linear layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    #complies the typography
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
    loss="mean_squared_error",
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

#trains the model
def train_model(model, df, feature, label, my_epochs, my_batch_size=None, my_validation_split=0.1):
    history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=my_batch_size,
                      epochs=my_epochs,
                      validation_split=my_validation_split)

  # Gather the model's trained weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the 
  # rest of history.
  epochs = history.epoch
  
  # Isolate the root mean squared error for each epoch.
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse, history.history   

print("Defined the build_model and train_model functions.")


#@title Define the plotting function

def plot_the_loss_curve(epochs, mae_training, mae_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
  plt.legend()
  
  # We're not going to plot the first epoch, since the loss on the first epoch
  # is often substantially greater than the loss for other epochs.
  merged_mae_lists = mae_training[1:] + mae_validation[1:]
  highest_loss = max(merged_mae_lists)
  lowest_loss = min(merged_mae_lists)
  delta = highest_loss - lowest_loss
  print(delta)

  top_of_y_axis = highest_loss + (delta * 0.05)
  bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

# The following variables are the hyperparameters.
learning_rate = 0.1
epochs = 30
batch_size = 20


#training the model
# Split the original training set into a reduced training set and a
# validation set. 
validation_split=0.3

# Identify the feature and the label.
my_feature="median_income"  # the median income on a specific city block.
my_label="median_house_value" # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based 
# solely on the neighborhood's median income.  

# Discard any pre-existing version of the model.
my_model = None
#to randomise the data
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

# Invoke the functions to build and train the model.
my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, 
                                    my_label, epochs, batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])



#use test datset to evaluate your models performance

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x-test, y_test, batch_size=batch_size)

