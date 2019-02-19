from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
trainpath = "E:\\MLtitanic\\data\\titanic_train.csv"
testpath = "E:\\MLtitanic\\data\\titanic_test.csv"
dftrain= pd.read_csv(trainpath)
dftest=pd.read_csv(testpath)

train = dftrain.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
test = dftest.drop(['Name','Ticket','Cabin'],axis=1)

alldata = [train,test]
for data in alldata:
    data['Sex']=data['Sex'].map({"female":1,'male':0})
    data['Embarked']=data['Embarked'].map({'C':1,'Q':2,'S':3})
    data.fillna(data.mean()['Age'],inplace=True)
    data.fillna(data.mean()['Fare'],inplace=True)
    data.fillna(method= 'pad')
    data.loc[(data['Age']<=16),'Age']=1
    data.loc[(data['Age'] >16)&(data['Age'] <=32), 'Age'] = 2
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 3
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 4
    data.loc[(data['Age'] > 64), 'Age'] = 5

# print(train[:20])
X_train = train.drop('Survived',axis=1)
Y_train = train['Survived']
X_testid = test['PassengerId']
X_test = test.drop('PassengerId',axis=1).copy()
#=================================================
classifiers=[DecisionTreeClassifier(),NuSVC(),XGBClassifier(),GradientBoostingClassifier(),LogisticRegression()]
paths=['DecisionTreeClassifier.csv','NuSVC.csv','XGBClassifier.csv','GradientBoostingClassifier.csv','LogisticRegression.csv']
for i in range (len(classifiers)):
    model = classifiers[i]
    model.fit(X_train,Y_train)
    Y_pred =pd.DataFrame(model.predict(X_test))
    from sklearn.metrics import accuracy_score
    result_csv = pd.concat([X_testid,Y_pred],axis=1)
    result_csv.columns=['PassengerId','Survived']
    print(result_csv[:2])
    result_csv.to_csv("data/"+paths[i],index =None)

