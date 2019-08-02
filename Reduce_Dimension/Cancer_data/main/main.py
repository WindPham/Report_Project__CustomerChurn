
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


class REDUCING(object):


    def __init__(self, file_name):
        self.__df = pd.read_csv(file_name);
        #print(self.__df.head(3));
        self.__drop_col(['id']);
        return;
    #=====================================================================================================================
    def reset(self):
        self.__df.reset_index(inplace=True);
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
        #print(self.__df.head(2));
        return;
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
        return;
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
        return;
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
        return;
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
        list_name_out = X.columns[indices];
        self.__drop_col(list_name_out);
        return;
    #=====================================================================================================================
    def forward_feature_selection(self, number_of_features_need_to_delete, label):
        Y = self.__df[label];
        self.__drop_col([label]);
        X = self.__df;
        X = pd.get_dummies(X);
        model = RandomForestClassifier(n_estimators = 100, max_depth = 100);
        var = [];
        for i in range(0,len(X.columns)-1):
            if ffs[0][i] >=10:
                var.append(X.columns[i])

        return;
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    #=====================================================================================================================
    














if __name__=='__main__':
    rd = REDUCING('data.csv');

    rd.missing_values_ratio(20.0);
    rd.reset();
    print(rd.shape())
    #rd.full_scatter();
    #rd.low_variance_filter(0.25);
    #rd.high_correlation_filter(0.8)
    #rd.random_forest(10, 'diagnosis');
    #rd.backward_feature_elimination(10, 'diagnosis');
    rd.forward_feature_selection(10, 'diagnosis')
    print(rd.shape())
    #print("f");