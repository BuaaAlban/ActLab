import pandas as pd
import numpy as np
trainpath = "E:\\MLtitanic\\data\\titanic_train.csv"
testpath = "E:\\MLtitanic\\data\\titanic_test.csv"
dftrain= pd.read_csv(trainpath)
dftest=pd.read_csv(testpath)

train = dftrain.drop(['Name','PassengerID','Ticket','Cabin'],axis=1)
test = dftest.drop(['Name','PassengerID','Ticket','Cabin'],axis=1)

alldata = [train,test]

