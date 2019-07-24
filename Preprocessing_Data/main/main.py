import MISSING as m
import NORMALIZING as norm
import OUTLIER_DETECTION as od
import CONVERTING as conv
import DISCRETISATION as discre
import AllLibrary as all
from AllLibrary import train_test_split
import support
import numpy as np
import INTEGRATION as inte
if __name__=="__main__":
    telecom_cust = all.pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv");
    telecom_cust.TotalCharges   = all.pd.to_numeric(telecom_cust.TotalCharges, errors='coerce');
    telecom_cust.MonthlyCharges = all.pd.to_numeric(telecom_cust.MonthlyCharges, errors='coerce');
    conv.CONVERTING().label(telecom_cust, 'Churn');
    df = telecom_cust.iloc[:,1:];
    m.MISSING().remove(df);

    Y = df['Churn'].copy();
    X = df.drop('Churn', axis = 1);

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2);
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.5);

    print("X train shape = (%d , %d)"%X_train.shape + ", Y train shape = (%d)"%Y_train.shape);
    print("X valid shape = (%d , %d)"%X_val.shape   + ", Y valid shape = (%d)"%Y_val.shape);
    print("X test  shape = (%d , %d)"%X_test.shape  + ", Y test  shape = (%d)"%Y_test.shape);

    X_train['Churn'] = Y_train;
    models = discre.DISCRETISATION().random_forest_all_cols(X_train,'Churn', Max_depth = 2, must_not_discre = ['SeniorCitizen']);
    X_val['Churn'] = Y_val;
    diff_set_val = discre.DISCRETISATION().support_other_set(models, X_val, X_train);
    X_test['Churn'] = Y_test;
    diff_set_test = discre.DISCRETISATION().support_other_set(models, X_test, X_train);

    X_train = X_train.drop("Churn", axis = 1);
    X_val = X_val.drop('Churn', axis = 1);
    X_test = X_test.drop('Churn', axis = 1);


    X_train = all.pd.get_dummies(X_train);
    X_val = all.pd.get_dummies(X_val);
    X_test = all.pd.get_dummies(X_test);
    list_missing_names_val  = np.setdiff1d(X_train.columns, X_val.columns);
    list_missing_names_tra  = np.setdiff1d(X_val.columns, X_train.columns);
    list_missing_names_tes = np.setdiff1d(X_train.columns, X_test.columns);

    inte.INTEGRATION().add_cols(X_val, list_missing_names_val, 0);
    inte.INTEGRATION().add_cols(X_test, list_missing_names_tes, 0);

    inte.INTEGRATION().entity_idenfication_column_not_similar(X_train, X_train);


    model = all.SVC(C = 50, kernel = 'rbf', gamma = 0.001);
    model.fit(X_val, Y_val);
    print(model.score(X_train, Y_train));




