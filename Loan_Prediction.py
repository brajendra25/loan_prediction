# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:26:51 2019

@author: Anjali Brajendra
"""

import pandas as pd

data_trained = pd.read_csv("E:/Brajendra/Docs-23-07-19/train_loan.csv")

#data_trained.head(10)

#data_trained.info()

#data_trained.shape

#data_trained['Loan_Status'].value_counts()

#data_trained['Credit_History'].value_counts()

#data_trained['Property_Area'].value_counts()

data_trained['Property_Area'] = data_trained['Property_Area'].map({'Semiurban':int('0'),'Urban':int('1'),'Rural':int('3')})

#data_trained['Property_Area'].value_counts()

data_trained['Education'].value_counts()

data_trained['Education'] = data_trained['Education'].map({'Graduate':int('0'),'Not Graduate':int('1')})

#data_trained['Education'].value_counts()

#data_trained

# Handling Missing Values

#data_trained.shape

data_na = data_trained.dropna()

#data_na.shape

data_na['Self_Employed']= data_na['Self_Employed'].map({'No':int('0'),'Yes':int('1')})

data_na['Married']= data_na['Married'].map({'No':int('0'),'Yes':int('1')})

data_na['Gender']= data_na['Gender'].map({'Male':int('0'),'Female':int('1')})

#data_na.info()


del data_na['Loan_ID']

data_na['Loan_Status']= data_na['Loan_Status'].map({'N':int('0'),'Y':int('1')})

#data_na

import matplotlib.pyplot as py
from sklearn import linear_model

reg  = linear_model.LinearRegression()
#reg.fit(data_na[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']],data_na.Loan_Status)

reg.fit(data_na[['ApplicantIncome','CoapplicantIncome','Credit_History']],data_na.Loan_Status)

#reg.coef_

#reg.intercept_

#reg.predict([[4583,0.0,0]])

data_test = pd.read_csv("E:/Brajendra/Docs-23-07-19/test_loan.csv")

#data_test

#data_test.info()

#data_test.shape

data_test['Property_Area'] = data_test['Property_Area'].map({'Semiurban':int('0'),'Urban':int('1'),'Rural':int('3')})
data_test['Education'] = data_test['Education'].map({'Graduate':int('0'),'Not Graduate':int('1')})
data_test['Self_Employed']= data_test['Self_Employed'].map({'No':int('0'),'Yes':int('1')})
data_test['Married']= data_test['Married'].map({'No':int('0'),'Yes':int('1')})
data_test['Gender']= data_test['Gender'].map({'Male':int('0'),'Female':int('1')})

del data_test['Loan_ID']

data_test.Gender = data_test.Gender.fillna(data_test.Gender.median())

data_test.Self_Employed = data_test.Self_Employed.fillna(data_test.Self_Employed.median())


data_test.Credit_History = data_test.Credit_History.fillna(data_test.Credit_History.median())

df = data_test[['ApplicantIncome','CoapplicantIncome','Credit_History']]
Row_list =[] 
  
# Iterate over each row 
for index, rows in df.iterrows(): 
    # Create list for the current row 
    my_list =[rows.ApplicantIncome, rows.CoapplicantIncome, rows.Credit_History] 
      
    # append the list to the final list 
    Row_list.append(my_list)


_prediction=[]
for items in Row_list:
    _pre=reg.predict([items])*100
    if _pre>60:
        _prediction.append('Y')
    else:
        _prediction.append('N')
    

reg.predict([[2137,8980,1]])*100

data_test['Loan_Status']=_prediction

data_test.info()

data_test.to_csv("E:/Brajendra/Docs-23-07-19/Prediction_loan.csv")
print("Loan Prediction Done !! Please chek this file : Prediction_loan.csv")