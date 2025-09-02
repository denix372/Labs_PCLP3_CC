import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
#matplotlib inline

train = pd.read_csv("train.csv")

train = train.set_index('PassengerId')

#load the test dataset
test = pd.read_csv('test.csv')

datadict = pd.DataFrame(train.dtypes)
datadict['MissingVal'] = train.isnull().sum()
datadict['NUnique'] = train.nunique()
datadict['Count'] = train.count()
datadict=datadict.rename(columns={0:'DataType'})
train.describe(include=['object'])
train.describe(include=['number'])
train.Survived.value_counts(normalize=True)


train['Name_len']=train.Name.str.len()
train['Ticket_First']=train.Ticket.str[0]
train['FamilyCount']=train.SibSp+train.Parch
train['Cabin_First']=train.Cabin.str[0]
# Regular expression to get the title of the Name
train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.title.value_counts().reset_index()
# mark zero values as missing or NaN
train.Fare = train.Fare.replace(0, np.NaN)

train[train.Fare.isnull()].index

# impute the missing Fare values with the mean Fare value
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

train['Age'] = train['Age'].fillna(train['Age'].mean())


trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]

# drop rows of missing values
trainML = trainML.dropna()

# Import Estimator AND Instantiate estimator class to create an estimator object
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

X_Age = trainML[['Age']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Age,y)

# Make a prediction
y_predict = lr.predict(X_Age)
y_predict[:10]

X_Fare = trainML[['Fare']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Fare,y)

# Make a prediction
y_predict = lr.predict(X_Fare)
y_predict[:10]

X_sex = pd.get_dummies(trainML['Sex']).values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_sex, y)
# Make a prediction
y_predict = lr.predict(X_sex)
y_predict[:10]

X_pclass = pd.get_dummies(trainML['Pclass']).values
y = trainML['Survived'].values
lr = LogisticRegression()
lr.fit(X_pclass, y)
# Make a prediction
y_predict = lr.predict(X_pclass)
y_predict[:10]

from sklearn.ensemble import RandomForestClassifier
X=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Name_len', 'FamilyCount']].values # Taking all the numerical values
y = trainML['Survived'].values
RF = RandomForestClassifier()
RF.fit(X, y)
# Make a prediction
y_predict = RF.predict(X)
y_predict[:10]
print((y == y_predict).mean())