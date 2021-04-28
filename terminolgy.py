#labels
#what we are trying to predict (email spam / not spam)

#features
#the input varibles used to predict the labels (the word count in an email, or the sender)
#represented as x1, x2

#labelled example, has the label and the feature

#the y label in simple linear re
#unlabelled example, has the features but not the label

#models are used to make the predictions (guess the label)

#example are instances of X , they can be labelled or unlabelled

#labelled example {features, label} : (x:y) are used to train the model. its a vector

#once trained, we use the model to predict unlabelled examples

#model defines the relationship between features and labels

#Inference means applying a trained model to unlabelled examples

#Two kinds of models
#regression gives a value (whats the average price of a house in UK)
#classification gives discrete values , spam not spam.. or is this a picture of a dog

#Overfitting, when the model fits too well on the trained data, but not on new data
#Ockhams razor ; the less comples a model is, the better it is as a model

#split the data into training and test. Train the model and then test is. If if works well on the test data, its not overfitting and therefore should work well on unseen data

#having two sets test and training, can still be overfitted, A solution is partitioning the data (the training data is split in Validation data and training) .
#the training is then evaluated on the validation data, then tweaked by the results of the test data
# pick the model with the lowest loss score on the test data

#string values as features get turning into 1 feature vectors with one-hot encoding

#representation is honing the model by imporving and adding features
#feature engineering is changing raw data into feature vectors

#categorical features have a discrete set of possible features (like street names)

#One-hot encoding extends to numeric data that you do not want to directly multiply by a weight, such as a postal code.

#Sparse representation is whenn the catergorical features are large but not all have values, therefore only the ones with values are stored

#discrete features should be used more than 5 times

#scaling means putting features in a 0 -1 range

#synthetic feature a feature cross is formed by multipling two or more features (longittue and rooms per person) x1 * x2 = x3

#cross features is used for more complex models

#overcrossing, when too many cross features in the model make it overfit to data, must simplfiy the model

#Regularization is used to stop overfitting (when training data does too good against the validation data)
#early stopping is used
#trying to stop model complexity
#while trainig

#l2 regulization
#penalizes model complexity
#so minimise loss and complexity

#to define model complexity (make the weights as small as possible)

#sum of the square value of weights
#penalizes big weights

#so two terms, the loss and regulisatiobn
#In other words, instead of simply aiming to minimize loss (empirical risk minimization):

#we'll now minimize loss+complexity, which is called structural risk minimization:
#tune the impact of L2 by using a scalar known as Lambda, or regulization rate .
#l2 encourages weight values twoards 0, and the mean of weights towards 0

#high Lambda values strengtens l2

#high values means a simple model, low values mean a more complex model
#setting lambda to 0 removes l2 regulization

#logistic regression
#when the probility is not 0 or 1, somewhere in the middle
#we need another way to log loss, logistic regression
#log loss

#The loss function for linear regression is squared loss. The loss function for logistic regression is Log Loss,
