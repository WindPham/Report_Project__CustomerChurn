import MISSING
import numpy as np

class OUTLIER_DETECTION(object):
    """description of class"""
    def __init__(self):
        return;
    def zScores_all_rows(self, df, Start = -2.5, End = 2.5):#df: dataframe --> pandas
        for i in df.columns:
            if df[i].dtypes != 'object':
                df["outlier"] = df[i].apply(lambda x: x <=Start or x >= End);
                outlier_indx = df[df.outlier==True].index.values;
                if (len(outlier_indx)>0):
                    df.iloc[outlier_indx,:]=np.nan;
                df.drop("outlier", axis=1, inplace=True);
            MISSING.MISSING().remove(df);
        return;

    def DBSCAN_all_rows(df, Eps = 2, Metric = 'euclidean', Min_samples = 5, N_jobs = -1):#df: dataframe --> pandas
        outlier_detection = DBSCAN(eps=Eps, metric=Metric, min_samples = Min_samples, n_jobs = N_jobs);
        df['outlier'] = outlier_detection.fit_predict(df);
        outlier_indx = df[df.outlier == 0].index.values;
        if (len(outlier_indx)>0):
            df.iloc[outlier_indx,:]=np.nan;
        df.drop("outlier", axis=1, inplace=True);
        MISSING.MISSING().remove(df);
        return;

    def IsolationRandomForest_all_rows(X, Y, Max_samples=10, # X: matrix, Y: matrix --> numpy
                                                   Random_state = np.random.RandomState(42), 
                                                   Contamination = 'auto', 
                                                   Behaviour = 'new'):
        clf = IsolationForest(max_samples=Max_samples, 
                              random_state=Random_state, 
                              contamination = Contamination, 
                              behaviour = Behaviour);
        clf.fit(X, Y);
        outliers = np.where(clf.predict(X) != -1);
        X = X[outliers];
        Y = Y[outliers];
        return;


