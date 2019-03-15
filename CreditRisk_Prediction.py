# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:26:17 2019

@author: g704874
"""

# Set working directory
import os
import pandas as pd
import numpy as np
os.getcwd()
os.chdir("C:/IMARTICUS Learning/PythonLearning/SVM_CreditRisk_UseCase")
os.getcwd()

####IMPORTING TRAINING AND TEST DATA SETS ######
TrainData = pd.read_csv("R_Module_Day_7.2_Credit_Risk_Train_data.csv")
TestData = pd.read_csv("R_Module_Day_8.2_Credit_Risk_Test_data.csv")

####CREATING A NEW COLUMN SOURCE UNDER BOTH TRAIN AND TEST DATA
TrainData["Source"] = "Train"
TestData["Source"] = "Test"

####COMBINE BOTH TRAIN AND TEST AS FULL DATA
FullData = pd.concat([TrainData,TestData])
FullData.shape

###View starting 5 records
FullData.head()

####Check the summary of Numerical variables
FullData.describe()

####Working on Categorical variable Dependents
FullData.Dependents.value_counts()   ###THere is an invalid category as 3+

FullData.Dependents = np.where(FullData.Dependents == '3+',3,FullData.Dependents).astype(float)
FullData.Dependents.value_counts()
FullData.Dependents.dtype

###Finding MISSING VALUES
FullData.isnull().sum()

## MISSING VALUE IMPUTATION
for col_name in list(FullData):
    if ((col_name not in ['Loan_ID', 'Loan_Status', 'Source']) & (FullData[col_name].isnull().sum() >0)):
        if(FullData[col_name].dtype != object):
            temp1 = FullData[col_name][FullData.Source == "Train"].median()
            FullData[col_name].fillna(temp1, inplace=True)
        else:
            temp2 = FullData[col_name][FullData.Source =="Train"].mode()[0]
            FullData[col_name].fillna(temp2, inplace=True)

FullData.isnull().sum()

###OUTLIER DETECTION AND CORRECTION
#ApplicantIncome
FullData[FullData.Source == "Train"].boxplot(column = 'ApplicantIncome')
FullData.ApplicantIncome.dtype
np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],[95,96,97,98,99])
        
FullData.ApplicantIncome = np.where(FullData.ApplicantIncome > np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],99),np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],99),FullData.ApplicantIncome)
FullData.ApplicantIncome = np.where(FullData.ApplicantIncome > np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],95),np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],95),FullData.ApplicantIncome)
FullData.ApplicantIncome = np.where(FullData.ApplicantIncome > np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],90),np.percentile(FullData.loc[FullData.Source == "Train","ApplicantIncome"],90),FullData.ApplicantIncome)

# CoapplicantIncome
FullData.columns
FullData[FullData.Source == "Train"].boxplot(column ="CoapplicantIncome")
np.percentile(FullData.loc[FullData.Source == "Train","CoapplicantIncome"],99)

FullData.CoapplicantIncome = np.where(FullData.CoapplicantIncome > np.percentile(FullData.loc[FullData.Source == "Train","CoapplicantIncome"],99),np.percentile(FullData.loc[FullData.Source == "Train","CoapplicantIncome"],99),FullData.CoapplicantIncome)
FullData.CoapplicantIncome = np.where(FullData.CoapplicantIncome > np.percentile(FullData.loc[FullData.Source == "Train","CoapplicantIncome"],95),np.percentile(FullData.loc[FullData.Source == "Train","CoapplicantIncome"],95),FullData.CoapplicantIncome)

# LoanAmount
FullData[FullData.Source == "Train"].boxplot(column ="LoanAmount")
np.percentile(FullData.loc[FullData.Source =="Train","LoanAmount"],99)
FullData.LoanAmount = np.where(FullData.LoanAmount > np.percentile(FullData.loc[FullData.Source =="Train","LoanAmount"],99),np.percentile(FullData.loc[FullData.Source=="Train","LoanAmount"],99),FullData.LoanAmount)
FullData.LoanAmount = np.where(FullData.LoanAmount > np.percentile(FullData.loc[FullData.Source =="Train","LoanAmount"],95),np.percentile(FullData.loc[FullData.Source=="Train","LoanAmount"],95),FullData.LoanAmount)
FullData.LoanAmount = np.where(FullData.LoanAmount > np.percentile(FullData.loc[FullData.Source =="Train","LoanAmount"],90),np.percentile(FullData.loc[FullData.Source=="Train","LoanAmount"],90),FullData.LoanAmount)

########ONE HOT ENCODING OF CATEGORICAL VARIABLES  BY CREATING DUMMY VARIABLES ########
cat = FullData.loc[:,FullData.dtypes == object].columns
Dummy = pd.get_dummies(FullData[cat].drop(['Loan_ID', 'Source', 'Loan_Status'], axis = 1),drop_first = True)
Dummy.shape
Dummy.columns

FullData2 = pd.concat([FullData,Dummy],axis =1)
FullData2.shape

Cols_To_Drop = ['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
FullData3 = FullData2.drop(Cols_To_Drop,axis = 1).copy()
FullData3.columns
FullData3.shape

# Convert Dependent variable into 0,1. If Loan_Status = N, then 1 else 0
FullData3.Loan_Status = np.where(FullData3.Loan_Status == 'N',1,0)
FullData3.Loan_Status.value_counts()
FullData3.shape
FullData3.dtypes

######SAMPLING #######################
# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column
Train = FullData3.loc[FullData3.Source == "Train",].drop("Source",axis = 1).copy()
Train.shape

Test = FullData3.loc[FullData3.Source == "Test",].drop("Source",axis =1).copy()
Test.shape

###DIVIDE EACH DATA SET AS INDEPENDENT AND DEPENDENT VARAIBLES
train_X = Train.drop("Loan_Status",axis = 1)
train_y = Train["Loan_Status"].copy()
test_X = Test.drop("Loan_Status",axis = 1)
test_y = Test["Loan_Status"].copy()


###################MODEL BUILDING ###############################
###SVM MODEL
from sklearn.svm import SVC
from sklearn.metrics import classification_report

M1 = SVC()
Model1 = M1.fit(train_X,train_y)
Pred1 = Model1.predict(test_X)

###CONFUSION MATRIX ##########################################
from sklearn.metrics import confusion_matrix
conf1 = confusion_matrix(test_y,Pred1)
print(conf1)     #### 79% accuracy
accuracy = ((conf1[0][0] + conf1[1][1])/test_y.shape[0]) * 100

report1 = classification_report(test_y,Pred1)
print(report1)

###Manual GRID Searches
Model_Validation_Df = pd.DataFrame()
mycost_List = []
mygamma_List = []
mykernel_List = []
accuracy_List = []
for mycost in [1,2]:
    for mygamma in [0.01, 0.1]:
        for mykernel in ['sigmoid','rbf']:
            Temp_Model = SVC(C = mycost, kernel = mykernel, gamma = mygamma)
            Temp_Model = Temp_Model.fit(train_X, train_y)
            Test_Pred = Temp_Model.predict(test_X)
            Confusion_Mat = confusion_matrix(test_y, Test_Pred)
            Temp_Accuracy = ((Confusion_Mat[0][0] + Confusion_Mat[1][1])/test_y.shape[0])*100
            print(mycost, mygamma, mykernel)
            print(Temp_Accuracy)
            print("******************************")
            mycost_List.append(mycost)
            mygamma_List.append(mygamma)
            mykernel_List.append(mykernel)
            accuracy_List.append(Temp_Accuracy)
            
Model_Validation_Df = pd.DataFrame({'Cost': mycost_List, 
                                    'Gamma': mygamma_List, 
                                    'Kernel': mykernel_List, 
                                    'Accuracy': accuracy_List})

















