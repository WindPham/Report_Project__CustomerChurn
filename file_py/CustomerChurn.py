import numpy as np
import scipy as scp
import sklearn as skl
import pandas as pd
import matplotlib
import pickle as pk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


#read data
print("Readind data");
telecom_cust = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv");

telecom_cust.TotalCharges = pd.to_numeric(telecom_cust.TotalCharges, errors='coerce')
telecom_cust.MonthlyCharges = pd.to_numeric(telecom_cust.MonthlyCharges, errors='coerce')


#removing missing data:
print("Remove the missind data:");
print("\tBefore removing: ", end='');
print(len(telecom_cust));
telecom_cust.dropna(inplace = True);
print("\tAfter removing: ", end='');
print(len(telecom_cust));


#normalizing by z-score method
def normalization_data_z_score(telecom_cust):
    print("Normalize by z-scores")
    for i in telecom_cust.columns:
        if telecom_cust[i].dtypes != 'object':
            meanX = telecom_cust[i].mean();
            sd = telecom_cust[i].std();
            telecom_cust[i] = (telecom_cust[i]-meanX)/sd;
normalization_data_z_score(telecom_cust);


#Remove customer IDs from the data set
df2 = telecom_cust.iloc[:,1:]


#Converting the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
label = df_dummies['Churn'].copy();
df_dummies.drop("Churn", axis=1, inplace = True);
df_dummies.to_csv("CustomerChurn.csv");

# Convert from dataframe to matrix
X = df_dummies.values;
Y = label.values;


#data set was divided: 5000:1000:1032 = train: Validation: test.
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2);
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15);
print("Divide date set:")
print("\tX train shape = (%d , %d)"%X_train.shape + ", Y train shape = (%d)"%Y_train.shape);
print("\tX valid shape = (%d , %d)"%X_val.shape + ", Y valid shape = (%d)"%Y_val.shape);
print("\tX test  shape = (%d , %d)"%X_test.shape + ", Y test  shape = (%d)"%Y_test.shape);


# dividing data set into three sets: training set (5000 records), validation set(1000 records), test set (1032 records)
model = SVC(C=50, kernel='rbf', gamma = 0.02);
out_model = model.fit(X_train, Y_train);
print("Score of accuracy: ", end='');
print(str(out_model.score(X_val,Y_val)*100) + "%");
