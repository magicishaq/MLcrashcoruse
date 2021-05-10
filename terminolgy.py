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

#Classification
#logicistic regression returns a probility . 1 being positive, 0 = negative. 0.9 is very likley while 0.03 is unlikley
#0.6? need a classification threshold /desicion threshold
#predictions in a confusion matrix, true positive: true prediction, true negative; false positive, false negative
#use classifcation models for these 4 outcomes

#accurancy = guessing how accurate our predictions are; accuracy =  num of correct predictions/ predictions
#accuracy of classifcation = Tp + TN / Tp+ TN + FP + FP

#class imbalanced dataset (loads of either negative or positive) is not good for measuring accuracy

#What proportion of positive identifications was actually correct? precision tell how many of the positives were correct = Precision = TP / TP + FP

#Recall: What proportion of actual positives was identified correctly? TP/ TP + FN

#raising classifcation will increase precision but lower recall

#decreasing the classifcation will increase false positives and false negatives decrease

#ROC curve ((receiver operating characteristic curve)) is a graph that shows the performance of a classification model at all classification thresholds
#The cure plots the true positive rate and flase positive rate

#TPR = TP/ TP + FN
#FPR = FP / FN + TN

#an ROC curve plots the TPR vs FPR at different thresholds. Lowering the classifcation theshold makes more item positive but will increase the false positives

#AUC: Area under the ROC curve , instead of measuring a logistic regession model many times with different classifcation threshols . use a sorting-based algorithm that can provide this information called AUC

#AUC measure the entire 2d space under the curve

#AUC is the probitlity a random positive is on the right side of the threshold

#CLassification: prediction Bias. Logisitic regression should be unbiased ; average of predictions should = average of observations

#prediction bias is the measure of how far apart the averages are

#causes of prediction bias: Incomplete feature set, nosiy data set, buggy pipline, biased training sample , overly strong regularization

#logisitic regression can olny classify things as 1 or 0. to find out the prediction bias, best to bucket the data into groups
