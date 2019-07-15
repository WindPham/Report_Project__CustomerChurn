import numpy as np
import scipy as scp
import sklearn as skl
import pandas as pd
import matplotlib
import pickle as pk
from sklearn.svm import SVC


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
df_dummies = pd.get_dummies(df2);
# Convert from dataframe to matrix
data_set = df_dummies.values
# dividing data set into three sets: training set (5000 records), validation set(1000 records), test set (1032 records)
def divide_data_set(data_set, k_train, k_val):
    R = data_set.shape[0];
    C = data_set.shape[1];
    rand_indx = np.random.choice(R, R, replace = True);
    rand_indx_train = rand_indx[np.arange(0,k_train)]
    rand_indx_val = rand_indx[np.arange(k_train,k_val + k_train)];
    rand_indx_test = rand_indx[np.arange(k_train + k_val,R)];
    #------------------------------------------------------------
    X_train = data_set[rand_indx_train];
    Y_train = X_train[:,-1:];
    X_train = X_train[:,0:C-1];
    print("X_train shape = (%d,%d)"%(X_train.shape));
    print("Y_train shape = (%d,%d)"%(Y_train.shape));
    #------------------------------------------------------------
    X_val = data_set[rand_indx_val];
    Y_val = X_val[:,-1:];
    X_val = X_val[:,0:C-1];
    #------------------------------------------------------------
    X_test = data_set[rand_indx_train];
    Y_test = X_test[:,-1:];
    X_test = X_test[:,0:C-1];
    return X_train, Y_train, X_val, Y_val, X_test, Y_test;
#data set was divided: 5000:1000:1032 = train: Validation: test.
X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_data_set(data_set, 5000,1000);

model = SVC(C=30, kernel='linear');
out_model = model.fit(X_train, Y_train);
print("Score of accuracy: ", end='');
print(out_model.score(X_val,Y_val)*100);
