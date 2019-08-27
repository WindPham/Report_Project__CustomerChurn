
import numpy as np
import pandas as pd 
import scipy as sp
import sklearn as skl
import time as time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import preprocessing
from sklearn.svm import SVC
class REDUCING(object):
    def __init__(self, file_name):
        self.__df = pd.read_csv(file_name);
        #print(self.__df.head(3));
        self.__drop_col(['id']);
        return;
    #=====================================================================================================================
    def reset(self):
        self.__df.reset_index(drop=True, inplace=True);
        return;
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    def full_scatter(self):
        for i in self.__df.columns:
            for j in self.__df.columns:
                if(i != j or i == j):
                    plt.scatter(self.__df[i], self.__df[j]);
                    plt.xlabel(i);
                    plt.ylabel(j);
                    plt.show();
    #=====================================================================================================================
    #=====================================================================================================================
    def __add_a_col(self, col_name, col):
        if(col_name not in self.__df.columns):
            self.__df[col_name]=col;
        return;
    #=====================================================================================================================
    def shape(self): return self.__df.shape;
    #=====================================================================================================================
    def __drop_col(self, list_names_out):
        self.__df.drop(list_names_out, axis=1, inplace = True);
        return;
    #=====================================================================================================================
    def missing_values_ratio(self, threshold):
        list_name_out = [];
        for i in self.__df.columns:
            if( (self.__df[i].isnull().sum() / self.__df.shape[0])*100 > threshold):
                list_name_out.append(i);
        self.__drop_col(list_name_out);
        X = self.__df;
        return X;
    #=====================================================================================================================
    def low_variance_filter(self, threshold):
        df_var = self.__df.var();
        list_name_out = [];
        for i in range(0, len(df_var.values)):
            if (df_var.values[i]*100 < threshold):
                #print(i);
                #print(self.__df.columns[i])
                #print(df_var.values[i]*100);
                #print('\n')
                list_name_out.append(self.__df.columns[i]);
        self.__drop_col(list_name_out);
        X = self.__df;
        return X;
    #=====================================================================================================================
    def high_correlation_filter(self, threshold):
        df_corr = self.__df.corr();
        list_out = []
        for i in range(0, len(df_corr.index)):
            for j in range(i+1, len(df_corr.columns)):
                if (df_corr.values[i][j] > threshold):
                    list_out.append(df_corr.columns[j]);
        list_name_out = list(set(list_out));
        self.__drop_col(list_name_out);
        X = self.__df;
        return X;
    #=====================================================================================================================
    def random_forest(self, number_of_features_need_to_delete, label):
        model = RandomForestClassifier(n_estimators = 100, max_depth = 100);
        Y = self.__df[label];
        self.__drop_col([label]);
        X=self.__df;
        X = pd.get_dummies(X);
        model.fit(X, Y);
        features = X.columns;
        important_features = model.feature_importances_;
        indices = np.argsort(important_features)[:number_of_features_need_to_delete-len(features)];
        #plt.title('Feature Importances')
        #plt.barh(range(len(indices)), important_features[indices], color='b', align='center')
        #plt.yticks(range(len(indices)), [features[i] for i in indices])
        #plt.xlabel('Relative Importance')
        #plt.show();
        list_name_out = features[indices];
        #for i in list_name_out:
        #    print(i);
        #print(important_features[indices])
        #print(np.sort(important_features))
        self.__drop_col(list_name_out);
        X = self.__df;
        return X
    #=====================================================================================================================
    def backward_feature_elimination(self, number_of_features_need_to_delete, label):
        perc = Perceptron();
        rfe = RFE(perc, 10);
        Y = self.__df[label];
        self.__drop_col([label]);
        X=self.__df;
        X = pd.get_dummies(X);
        rfe.fit_transform(X, Y);
        rank = rfe.ranking_;
        #print(rank);
        indices = np.argsort(rank)[:number_of_features_need_to_delete-len(X.columns)];
        print(indices)
        list_name_out = X.columns[indices];
        self.__drop_col(list_name_out);
        X = self.__df;
        return X;
    #=====================================================================================================================
    def forward_feature_selection(self, number_of_features_need_to_delete, label):
        Y = self.__df[label];
        self.__drop_col([label]);
        X = self.__df;
        X = pd.get_dummies(X);
        model = RandomForestClassifier(n_estimators = 100, max_depth = 100);
        reduced_model = sfs(model, k_features = len(X.columns) - number_of_features_need_to_delete, 
                            forward = True, floating = False, verbose = 2, scoring = 'accuracy', cv = 5);
        reduced_model.fit(X, Y);
        feature_cols = list(reduced_model.k_feature_idx_)
        list_name_out = list(X.columns[feature_cols]);
        self.__drop_col(list_name_out);
        X = self.__df;
        return X;
    #=====================================================================================================================
    def factor_analysis(self, number_of_features_need_to_delete, label):
        fa = FactorAnalyzer(rotation = None);
        Y = self.__df[label];
        self.__drop_col([label]);
        X = self.__df;
        X = pd.get_dummies(X);
        fa.fit(X, len(X.columns)-number_of_features_need_to_delete)
        
        val = fa.get_communalities();
        indices = list(np.argsort(val))[:len(X.columns)-number_of_features_need_to_delete];
        list_name_out = X.columns[indices];
        self.__drop_col(list_name_out)
        X = self.__df;
        return X;
    #=====================================================================================================================
    def principle_component_analysis(self, number_of_features_need_to_delete, label):
        Y = self.__df[label];
        self.__drop_col([label]);
        X = self.__df;
        X = pd.get_dummies(X);
        pca_model = PCA(n_components = len(X.columns)-number_of_features_need_to_delete);
        X = preprocessing.scale(X);
        pca_model.fit(X);
        X = pca_model.transform(X);
        return pca_model, X, np.array(Y);
    #=====================================================================================================================
    def independent_component_analysis(self, number_of_features_need_to_delete, label):
        Y = self.__df[label];
        self.__drop_col([label]);
        X = self.__df;
        X = pd.get_dummies(X);
        ica_model = FastICA(n_components = len(X.columns)-number_of_features_need_to_delete);
        X = preprocessing.scale(X);
        ica_model.fit(X);
        X = ica_model.transform(X);
        return ica_model, X, np.array(Y);
    #=====================================================================================================================
def make_label_out(df, label):
    Y = df[label];
    df.drop([label], axis = 1, inplace = True);
    X = df;
    X=preprocessing.scale(X);
    return np.array(X), np.array(Y);
def model_make_df(model, X):
    return model.transform(X);

def test_for_reducing():
    rd = REDUCING('data_train.csv');
    val = pd.read_csv('data_val.csv');
    val.reset_index(drop=True, inplace=True);
    test = pd.read_csv('data_test.csv');
    test.reset_index(drop=True, inplace=True);
    rd.missing_values_ratio(20.0);
    rd.reset();
    #print(rd.shape())
    #rd.full_scatter();
    #rd.low_variance_filter(0.25);
    #rd.high_correlation_filter(0.8);
    #rd.random_forest(10, 'diagnosis');
    #rd.backward_feature_elimination(10, 'diagnosis');
    #rd.factor_analysis(20, 'diagnosis');
    #X, Y = rd.principle_component_analysis(10, 'diagnosis');
    model, X, Y = rd.principle_component_analysis(15, 'diagnosis');
    Xv, Yv = make_label_out(val, 'diagnosis');
    Xt, Yt = make_label_out(test, 'diagnosis');

    Xv=model_make_df(model, Xv);
    Xt=model_make_df(model, Xt);
    print(X.shape);
    print(Y.shape);
    print(Xv.shape);
    print(Yv.shape);
    print(Xt.shape);
    print(Yt.shape);

    Cs = [0.0001, 0.000945, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.009, 0.01, 0.02, 0.04, 0.05, 0.075, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 2, 5, 10, 20, 50, 100];
    for i in Cs:
        svm_model = SVC(kernel='linear', C = i);
        svm_model.fit(X, Y);
        tr_sc = svm_model.score(X, Y);
        va_sc = svm_model.score(Xv, Yv);
        print("Reducing set: C = %f: kernel = linear: train-score = %f: val-score = %f"%(i,tr_sc,va_sc));
def test():
    train = pd.read_csv('data_train.csv');
    train.drop('id', axis=1, inplace=True);
    train.reset_index(drop=True, inplace=True);
    val = pd.read_csv('data_val.csv');
    val.reset_index(drop=True, inplace=True);
    test = pd.read_csv('data_test.csv');
    test.reset_index(drop=True, inplace=True);

    X, Y = make_label_out(train, 'diagnosis');
    Xv, Yv = make_label_out(val, 'diagnosis');
    Xt, Yt = make_label_out(test, 'diagnosis');

    print(X.shape);
    print(Y.shape);
    print(Xv.shape);
    print(Yv.shape);
    print(Xt.shape);
    print(Yt.shape);

    Cs = [0.0001, 0.000945, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.009, 0.01, 0.02, 0.04, 0.05, 0.075, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 2, 5, 10, 20, 50, 100];
    for i in Cs:
        svm_model = SVC(kernel='linear', C = i);
        svm_model.fit(X, Y);
        tr_sc = svm_model.score(X, Y);
        va_sc = svm_model.score(Xv, Yv);
        print("Normal set: C = %f: kernel = linear: train-score = %f: val-score = %f"%(i,tr_sc,va_sc));
if __name__=='__main__':
    test_for_reducing();
    test();