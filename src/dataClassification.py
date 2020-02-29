from random import randint
import scipy.io
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix


mat = scipy.io.loadmat('../data/data.mat')

data = mat["data"]

headers = mat["__header__"]
globals_V = mat["__globals__"]

samples = data[0][0][0]
labels = data[0][0][1]

#print(type(mat))
#print(mat.keys())
#print(type(data))

#print(headers)
#print(globals_V)

print("samples Shape: " + str(samples.shape))
#print("samples Size: " + str(samples.size))

print("labels Shape: " + str(labels.shape))
#print("labels Size: " + str(labels.size))

random_state = randint(0,9999)
#print("Random: " + str(random_state))

predicted_labels_vector = []
test_labels_vector = []

numberofFold = 5 # Equals to the number of TestSet

randomKfold = RepeatedKFold(n_splits=numberofFold, n_repeats=1, random_state=random_state)

for train_index, test_index in randomKfold.split(labels):
   print("%s %s" % (train_index.size, test_index.size))
   train_data = samples[train_index]
   train_labels = labels[train_index]

   test_data = samples[test_index]
   test_labels = labels[test_index]
   
   #print("Training Data Shape: " + str(train_data.shape))
   #print("Training Labels Shape: " + str(train_labels.ravel().shape))

   linear_svc = SVC(kernel='linear')
   linear_svc.fit(train_data,train_labels.ravel())
   predict_labels = linear_svc.predict(test_data)
   
   predicted_labels_vector = np.concatenate((predicted_labels_vector,predict_labels))
   test_labels_vector =  np.concatenate((test_labels_vector,test_labels.ravel()))

CM = confusion_matrix(test_labels_vector, predicted_labels_vector,normalize='all')
True_Negative = CM[0,0];
True_Positive = CM[1,1];
False_Negative = CM[1,0];
False_Positive = CM[0,1];

Accuracy = (True_Positive + True_Negative)/labels.size * 100;
Sensitivity = (True_Positive)/(True_Positive + False_Negative) * 100;
Specificity = (True_Negative)/(True_Negative + False_Positive) * 100;

print("Accuracy: " + str(Accuracy))
print("Sensitivity: " + str(Sensitivity))
print("Specificity: " + str(Specificity))