import pandas as pd
import seaborn as sb
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn.cross_validation import StratifiedKFold
sb.set

file_path_train ="/Users/jvr605/Downloads/kaggle_titanin_dataset/train.csv" 
file_path_test="/Users/jvr605/Downloads/kaggle_titanin_dataset/test.csv"
file_path_results="/Users/jvr605/Downloads/kaggle_titanin_dataset/titanic_results.csv"

df = pd.read_csv(file_path_train)
df.Age=df.Age.fillna(value = df.Age.mean())
#removing cabin details from the data frame
df = df.drop('Cabin',axis=1)
#Name column is not so useful,so removing it from the data frame
df  = df.drop('Name',axis=1)
#Ticket number does not make any sense 
df =df.drop('Ticket',axis=1)
#Conceriting the Embarked column catergorical data into the Numerical data 
df.Embarked[df.Embarked =='C'] = 1
df.Embarked[df.Embarked =='S'] = 2
df.Embarked[df.Embarked =='Q'] = 3
df.Embarked = df.Embarked.fillna(2)
""" Perfect there is no Missing data :) """
#Converiting the Female =1 and Male =2 #

df.Sex[df.Sex=='female'] = 1
df.Sex[df.Sex=='male']   = 2

"""Data is cleaned up ,and there no null/NAN  values 
Its time to perform Exploratory data analyis the data frome"""


train_columns= ['Pclass','Sex','Age',
                'SibSp','Parch','Fare','Embarked']
test_columns = ['Survived']
#Considering the inputs for training and testing values to validate against
X =df.loc[:,train_columns].values
Y =df.loc[:,test_columns].values



X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,random_state=0)

clf = ensemble.GradientBoostingClassifier(n_estimators=104)
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))


"""#Perfoming  validation on test.csv from Kaggle"""
df_test = pd.read_csv(file_path_test)
df_test.Age=df_test.Age.fillna(value = df_test.Age.mean())
#removing cabin details from the data frame
df_test = df_test.drop('Cabin',axis=1)
#Name column is not so useful,so removing it from the data frame
df_test  = df_test.drop('Name',axis=1)
#Ticket number does not make any sense 
df_test =df_test.drop('Ticket',axis=1)
#Conceriting the Embarked column catergorical data into the Numerical data 
df_test.Embarked[df_test.Embarked =='C'] = 1
df_test.Embarked[df_test.Embarked =='S'] = 2
df_test.Embarked[df_test.Embarked =='Q'] = 3
df_test.Embarked = df_test.Embarked.fillna(2)
""" Perfect there is no Missing data :) """
#Converiting the Female =1 and Male =2 #

df_test.Sex[df_test.Sex=='female'] = 1
df_test.Sex[df_test.Sex=='male']   = 2

df_test.Fare = df_test.Fare.fillna(df_test.Fare.median())

predict_Values = clf.predict(df_test.loc[:,train_columns].values)

"""preparing the file to upload into kaggle """
output = pd.DataFrame({'PassengerId':df_test.loc[:,'PassengerId'].values,'Survived':predict_Values})
output.to_csv(file_path_results,index=False)