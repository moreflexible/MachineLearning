import numpy as np
import enum 
import scipy.io
from random import randint
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

class CV_Model(enum.Enum): 
    LOO = 1
    K5 = 2
    K10 = 3

def ReadDataSetFromFile(filename):
   matFile = scipy.io.loadmat(filename)

   data = matFile["data"]
   #headers = matFile["__header__"]
   #lobals_V = matFile["__globals__"]

   samples = data[0][0][0]
   labels = data[0][0][1]

   #print("samples Shape: " + str(samples.shape))
   #print("labels Shape: " + str(labels.shape))

   return [samples, labels]

def ReportAccuracyResults(results): 
   print("********************************")
   print("Accuracy: " + str(results[0]))
   print("Sensitivity: " + str(results[1]))
   print("Specificity: " + str(results[2]))

def SelectCVModel(typeFold): 
   random_state = randint(0,100000)
   if typeFold == CV_Model.LOO:
      return LeaveOneOut()
   elif typeFold == CV_Model.K5:
      return  RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
   else:
      return RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)

def TrainAndTest(foldType, X, Y): 
   predicted_labels_vector = []
   test_labels_vector = []
   
   CVModel = SelectCVModel(foldType)

   for train_index, test_index in CVModel.split(Y):
      #print("%s %s" % (train_index.size, test_index.size))
      train_data = X[train_index]
      train_labels = Y[train_index]

      test_data = X[test_index]
      test_labels = Y[test_index]
      
      #print("Training Data Shape: " + str(train_data.shape))
      #print("Training Labels Shape: " + str(train_labels.ravel().shape))

      linear_svc = SVC(kernel='linear')
      linear_svc.fit(train_data,train_labels.ravel())
      predict_labels = linear_svc.predict(test_data)
      
      predicted_labels_vector = np.concatenate((predicted_labels_vector,predict_labels))
      test_labels_vector =  np.concatenate((test_labels_vector,test_labels.ravel()))

   CM = confusion_matrix(test_labels_vector, predicted_labels_vector,normalize='all')

   True_Negative = CM[0,0]
   True_Positive = CM[1,1]
   False_Negative = CM[1,0]
   False_Positive = CM[0,1]

   Accuracy = (True_Positive + True_Negative)/Y.size * 100
   Sensitivity = (True_Positive)/(True_Positive + False_Negative) * 100
   Specificity = (True_Negative)/(True_Negative + False_Positive) * 100

   return [Accuracy, Sensitivity, Specificity]

