import pandas as pd
import numpy as np
import seaborn as sb
trainpath = "E:\\MLtitanic\\data\\titanic_train.csv"
testpath = "E:\\MLtitanic\\data\\titanic_test.csv"
df= pd.read_csv(trainpath)
dftest=pd.read_csv(testpath)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width',2000)
pd.set_option('display.max_row',30)
pd.set_option('display.max_column',50)
print("关键词有哪些特征")
print(df.keys())
print(df.head())
keys=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']
print("类别信息统计")
print(df.describe())
print("训练集数据类型和缺失情况")
'''
共868条数据
Cabin          196 non-null object， 删除
Age            694 non-null float64，补充
Embarked       867 non-null object，补充
'''
print(df.info())
print("测试集数据类型和缺失情况")
'''
共410条数据
Age            327 non-null float64
Fare           409 non-null float64
Cabin          90 non-null object
'''
print(dftest.info())
#分析不同属性进行分组，查看与存活间的关系===========================
for feature in keys[:-1]:
    temp=df[[feature,'Survived']].groupby([feature],as_index=False).mean().sort_values(by="Survived",ascending=False)
    print(temp)
#数据之间的相关性================================================================
corr_matrix = df.corr()
print("数据之间的相关性")
print(corr_matrix["Survived"].sort_values())

print("存活情况：",df["Survived"].value_counts())


