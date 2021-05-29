import pandas
import numpy
import math

pathXTrain = 'x_train.csv'
dfXTrain = pandas.read_csv(pathXTrain)

pathYTrain = 'y_train.csv'
dfYTrain = pandas.read_csv(pathYTrain)

pathXTest = 'x_test.csv'
pathYTest = 'y_test.csv'
dfXTest = pandas.read_csv(pathXTest)
dfYTest = pandas.read_csv(pathYTest)

pathVocabulary = 'vocabulary.txt'
vocab = numpy.loadtxt(pathVocabulary, dtype=str)

# Multinominal Naive Bayes Classifier
numOfSpam = dfYTrain.value_counts().to_numpy()[0]
numOfNonSpam = dfYTrain.value_counts().to_numpy()[1]
priorSpam = numOfSpam / (numOfSpam + numOfNonSpam)
priorNonSpam = numOfNonSpam / (numOfSpam + numOfNonSpam)
loggedPriorSpam = math.log(priorSpam)
loggedPriorNonSpam = math.log(priorNonSpam)

alpha = 1

labelCounts = dfYTrain.value_counts().to_numpy() # Y = [1, 0], count  = [2910, 1174]

X = dfXTrain.to_numpy()
Y = dfYTrain.to_numpy()
test_X = dfXTest.to_numpy()
transposed_X = X.T

T_j_given_y_spam = (numpy.matmul(transposed_X, Y)) + alpha 
sum_T_j_given_y_spam = numpy.sum(T_j_given_y_spam) + alpha * vocab.shape[0]
P_X_given_Y_spam = T_j_given_y_spam / sum_T_j_given_y_spam

temp = 1 - Y
T_j_given_y_normal = numpy.matmul(transposed_X, temp) + alpha
sum_T_j_given_y_normal = numpy.sum(T_j_given_y_normal) + alpha * vocab.shape[0]
P_X_given_Y_normal = T_j_given_y_normal / sum_T_j_given_y_normal

logged_P_X_given_Y_spam = numpy.log(P_X_given_Y_spam)
logged_P_X_given_Y_normal = numpy.log(P_X_given_Y_normal)

tiles_probability = (numpy.tile(logged_P_X_given_Y_spam, test_X.shape[0])).T
element_wise_mult = numpy.multiply(test_X, tiles_probability)
spam = loggedPriorSpam + numpy.nansum(element_wise_mult, axis=1)

tiles_probability = (numpy.tile(logged_P_X_given_Y_normal, test_X.shape[0])).T
element_wise_mult = numpy.multiply(test_X, tiles_probability)
normal = loggedPriorNonSpam + numpy.nansum(element_wise_mult, axis=1)

predictions = (spam > normal).astype(int)

tn = 0
fn = 0
fp = 0
tp = 0
test_labels = dfYTest.to_numpy().flatten()
for i in range(len(predictions)):
  if (predictions[i] == 1 and test_labels[i] == 1):
    tp += 1
  elif (predictions[i] == 0 and test_labels[i] == 0):
    tn += 1
  elif (predictions[i] == 1 and test_labels[i] == 0):
    fp += 1
  elif (predictions[i] == 0 and test_labels[i] == 1):
    fn += 1
print(numpy.array([[tp, fp], [fn, tn]]))
print((tp + tn)/ (tp + tn + fn + fp))

# Bernoulli Naive Bayes
X = dfXTrain.to_numpy()
X[X > 0] = 1
Y = dfYTrain.to_numpy()
test_X = dfXTest.to_numpy()

# Prioris
numOfSpam = dfYTrain.value_counts().to_numpy()[0]
numOfNonSpam = dfYTrain.value_counts().to_numpy()[1]
priorSpam = numOfSpam / (numOfSpam + numOfNonSpam)
priorNonSpam = numOfNonSpam / (numOfSpam + numOfNonSpam)
# Likelihoods

# Spam
spam_emails = numpy.matmul(Y.T, X)
sum_columns_X_spam = numpy.sum(spam_emails, axis=0)
theta_j_spam = sum_columns_X_spam / numOfSpam
theta_j_spam = numpy.reshape(theta_j_spam, (theta_j_spam.shape[0], 1))

# Normal
normal_emails = numpy.matmul(1 - Y.T, X)
sum_columns_X_normal = numpy.sum(normal_emails, axis=0)
theta_j_normal = sum_columns_X_normal / numOfNonSpam
theta_j_normal = numpy.reshape(theta_j_normal, (theta_j_normal.shape[0], 1))

normal_probability_matrix = (numpy.tile(theta_j_normal, test_X.shape[0])).T
spam_probability_matrix = (numpy.tile(theta_j_spam, test_X.shape[0])).T

# Normal
first_term = numpy.multiply(normal_probability_matrix, test_X)
second_term = numpy.multiply(1 - normal_probability_matrix, 1 - test_X)
total = first_term + second_term
logged_total = numpy.log(total)
finalNormal = numpy.nansum(logged_total, axis=1)

# Spam
first_term = numpy.multiply(spam_probability_matrix, test_X)
second_term = numpy.multiply(1 - spam_probability_matrix, 1 - test_X)
total = first_term + second_term
logged_total = numpy.log(total)
finalSpam = numpy.nansum(logged_total, axis=1)

tn = 0
fn = 0
fp = 0
tp = 0
test_labels = dfYTest.to_numpy().flatten()
predictions = finalSpam > finalNormal
for i in range(len(predictions)):
  if (predictions[i] == 1 and test_labels[i] == 1):
    tp += 1
  elif (predictions[i] == 0 and test_labels[i] == 0):
    tn += 1
  elif (predictions[i] == 1 and test_labels[i] == 0):
    fp += 1
  elif (predictions[i] == 0 and test_labels[i] == 1):
    fn += 1
print(numpy.array([[tp, fp], [fn, tn]]))
print((tp + tn)/ (tp + tn + fn + fp))