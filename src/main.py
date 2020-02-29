from dataClassification import *

def main():

   samples,labels = ReadDataSetFromFile('../data/data.mat')
   #uniqueValues, occurCount = np.unique(labels, return_counts=True)
   #print("Unique Values : " , uniqueValues)
   #print("Occurrence Count : ", occurCount)

   Results_K5 = TrainAndTest(CV_Model.K5,samples,labels)
   ReportAccuracyResults(Results_K5)

   Results_K10 = TrainAndTest(CV_Model.K10,samples,labels)
   ReportAccuracyResults(Results_K10)

   Results_LOO = TrainAndTest(CV_Model.LOO,samples,labels)
   ReportAccuracyResults(Results_LOO)

if __name__== "__main__" :
   main()