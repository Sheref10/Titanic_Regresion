import numpy as np
import pandas as pd
import matplotlib.pyplot as ptl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
#Data collection
data=pd.read_csv('Survived_Train.csv')
data_test=pd.read_csv('Survived_Test.csv')
data_actual=pd.read_csv('Survived_TestActualOutput.csv')

#Data Preparation
count=0
for i in data.isnull().sum(axis=1):
  if i>0:
   count=count+1
print('Total number of rows with missing values is',count)
print('since it is only',round((count/len(data.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')

data['Title']=data['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
data_test['Title']=data_test['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())

data['Age'].fillna(data.groupby('Title')['Age'].transform('median'),inplace=True)
data_test['Age'].fillna(data.groupby('Title')['Age'].transform('median'),inplace=True)
#print('data[Age]=null',data['Age'].isnull().sum())
#print('data_test[Age]=null',data_test['Age'].isnull().sum())
DropFeauters=['PassengerId','Name','Ticket','Cabin','Embarked','Title','Fare']
data=data.drop(DropFeauters,axis=1)
data_test=data_test.drop(DropFeauters,axis=1)

#Feature Engineering
data['Sex']=data['Sex'].map({'male':1,'female':0})
data_test['Sex']=data_test['Sex'].map({'male':1,'female':0})
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
data['FamilyType'] = data['FamilySize'].map(lambda s: 1 if s == 1 else (2 if s >= 2 and s <= 4 else 3))
data_test['FamilySize'] = data_test['Parch'] + data_test['SibSp'] + 1
data_test['FamilyType'] = data_test['FamilySize'].map(lambda s: 1 if s == 1 else (2 if s >= 2 and s <= 4 else 3))

dropped_features = ['SibSp', 'Parch', 'FamilySize']
data=data.drop(dropped_features,axis=1)
data_test=data_test.drop(dropped_features,axis=1)

#Modeling
X_train=data.iloc[:,1:]
y_train=data.iloc[:,0]
X_test=data_test
y_acutal=data_actual['Survived'].values

model=LogisticRegression()
model.fit(X_train,y_train)
y_pre=model.predict(X_test)
#print(y_pre)

# Accuracy
cm = confusion_matrix(y_acutal, y_pre)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
#################
TN = cm[0, 0]
TP = cm[1, 1]
FN = cm[1, 0]
FP = cm[0, 1]
sensitivity = TP / float(TP + FN)
specificity = TN / float(TN + FP)
print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ', (TP + TN) / float(TP + TN + FP + FN), '\n',
      'The Missclassification = 1-Accuracy = ', 1 - ((TP + TN) / float(TP + TN + FP + FN)), '\n',
      'Sensitivity or True Positive Rate = TP/(TP+FN) = ', TP / float(TP + FN), '\n',
      'Specificity or True Negative Rate = TN/(TN+FP) = ', TN / float(TN + FP), '\n',
      'Positive Predictive value = TP/(TP+FP) = ', TP / float(TP + FP), '\n',
      'Negative predictive Value = TN/(TN+FN) = ', TN / float(TN + FN), '\n',
      'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ', sensitivity / (1 - specificity), '\n',
      'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ', (1 - sensitivity) / specificity)
