# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data
2.Define your model
3.Define your cost function
4.Define your learning rate
5.Train your model
6.Evaluate your model
7.Tune hyperparameters
8.Deploy your model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ENBANATHAN V
RegisterNumber:  212224220027

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![Screenshot 2025-05-18 125355](https://github.com/user-attachments/assets/c9c2c2fe-3dfc-475e-b424-2349e9a99915)
![Screenshot 2025-05-18 125409](https://github.com/user-attachments/assets/efa34c3e-f38b-4930-97fa-1b6ccd878870)
![Screenshot 2025-05-18 125425](https://github.com/user-attachments/assets/4fa0e324-56bb-40b0-99f7-8864e6081347)
![Screenshot 2025-05-18 125438](https://github.com/user-attachments/assets/c97f7572-6c47-4dcc-b70a-3739d899319d)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
