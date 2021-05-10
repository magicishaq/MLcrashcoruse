#linear regression is finding a straight line that best fits a set of points

#when drawing a line. Loss shows the distance betweent the examples and the line

#prediction of a value, and the true value (the difference)
#loss is always on a 0 - positive scale


#loss regression l2 loss (mean squared loss) 
#(observation - prediction)^2


#training a model to avoid loss is called empirical risk minimization
#loss is a penalty for a bad prediction
#mean squared loss is the (observation - prediction)^2 of all the examples mean average
#loss function l2 loss 


#hyper parameters are used to config the training model 


#learning rate is a hyperparater that tells the model how much to jump the gradient  in order to minimse loss. The gradient is the Weight

# y = b + w^1x^1 ; y = unlabel b = bais , w = weight (angle), x = features
# 
# when training loss has reached the lowest point the model has converged

#  iterative approach is adjusting the weight and bias by trail and using a loss function to adjust the loss in the right direction to minimise it

# Stochastic gradient descent (SGD) is taking one small batch from the data (at random) to let the model converged

#a batch size is the amount of examples
#a n ephoc is the when the model goes through the entire dataset

#to help a model converge
#training loss should decrease over time
#if the model does not converge, train for more epochs
#if the loss decreases too slowly, increase the learning rate
#if the training loss varies wildy, decrease the learning rate
#try a large batch size then decrease
#for large data sets, reduce the batch size to fit into the model



#A good dataset
#stored in csv format
#column names in the first row


