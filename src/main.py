from dataClassification import *
import numpy as np

def CreateUnbalanceDataset(X,Y):
   unbalanced_samples = X[0:40]
   unbalanced_samples = np.concatenate((unbalanced_samples,X[44:64]))

   unbalanced_labels = Y[0:40]
   unbalanced_labels = np.concatenate((unbalanced_labels,Y[44:64]))

   return [unbalanced_samples, unbalanced_labels]

def main():
   print("CV Comparison on Balanced Dateset")
   samples,labels = ReadDataSetFromFile('../data/data.mat')
   uniqueValues, occurCount = np.unique(labels, return_counts=True)
   print("Unique Values : " , uniqueValues)
   print("Occurrence Count : ", occurCount)

   Results_K5 = TrainAndTest(CV_Model.K5,samples,labels)
   ReportAccuracyResults(Results_K5)

   Results_K10 = TrainAndTest(CV_Model.K10,samples,labels)
   ReportAccuracyResults(Results_K10)

   Results_LOO = TrainAndTest(CV_Model.LOO,samples,labels)
   ReportAccuracyResults(Results_LOO)

   print("\nCV Comparison on Unbalanced Dateset")

   unbalanced_samples, unbalanced_labels = CreateUnbalanceDataset(samples, labels)

   uniqueValues2, occurCount2 = np.unique(unbalanced_labels, return_counts=True)
   print("Unique Values : " , uniqueValues2)
   print("Occurrence Count : ", occurCount2)

   uResults_K5 = TrainAndTest(CV_Model.K5,unbalanced_samples,unbalanced_labels)
   ReportAccuracyResults(uResults_K5)

   uResults_K10 = TrainAndTest(CV_Model.K10,unbalanced_samples,unbalanced_labels)
   ReportAccuracyResults(uResults_K10)

   uResults_LOO = TrainAndTest(CV_Model.LOO,unbalanced_samples,unbalanced_labels)
   ReportAccuracyResults(uResults_LOO)

if __name__== "__main__" :
   main()