import MISSING as m
import NORMALIZING as norm
import OUTLIER_DETECTION as od
import CONVERTING as conv
import DISCRETISATION as discre
import AllLibrary as all
from AllLibrary import train_test_split
if __name__=="__main__":
    telecom_cust = all.pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv");
    telecom_cust.TotalCharges   = all.pd.to_numeric(telecom_cust.TotalCharges, errors='coerce');
    telecom_cust.MonthlyCharges = all.pd.to_numeric(telecom_cust.MonthlyCharges, errors='coerce');
    conv.CONVERTING().label(telecom_cust, 'Churn');
    df = telecom_cust.iloc[:,1:];
    m.MISSING().remove(df);
    List = list(df);
    #discre.DISCRETISATION().decision_tree(df, 'tenure', 'Churn', Max_depth = 2);
    discre.DISCRETISATION().decision_tree_all_cols(df, 'Churn', Max_depth = 6, must_not_discre = ['SeniorCitizen']);

    df = all.pd.get_dummies(df);


    Y = df['Churn'].copy();
    X = df.drop('Churn', axis = 1);
    print(X.dtypes);

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2);
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25);

    print("X train shape = (%d , %d)"%X_train.shape + ", Y train shape = (%d)"%Y_train.shape);
    print("X valid shape = (%d , %d)"%X_val.shape + ", Y valid shape = (%d)"%Y_val.shape);
    print("X test  shape = (%d , %d)"%X_test.shape + ", Y test  shape = (%d)"%Y_test.shape);

    model = all.SVC(C = 50, kernel = 'rbf', gamma = 0.001);
    model.fit(X_train, Y_train);
    print(model.score(X_val, Y_val));




