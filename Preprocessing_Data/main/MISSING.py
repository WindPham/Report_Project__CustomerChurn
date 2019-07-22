import numpy as np
import scipy as scp
import sklearn as skl
import pandas as pd
import matplotlib
import pickle as pk
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split 

class MISSING(object):
    """description of class"""
    def __init__(self):
        return;
    def remove(self, df):
        df.dropna(inplace=True);
        df.reset_index(drop=True,inplace = True);
        return;
    def mean(self, df):
        list_name_of_attributes = df.columns;
        for i in list_name_of_attributes:
            #print(df[i].value_counts().index);
            if( df[i].isnull().sum()>0 and df[i].dtypes != 'object'):
                df[i].fillna(df[i].mean(), inplace=True);
            else:
                if (df[i].isnull().sum()>0 and df[i].dtypes == 'object'):
                    #print(df[i].value_counts().index);
                    df[i].fillna(df[i].value_counts().index.sort_values()[0], inplace=True);
        return;
    #---------------------------------------------------------------
    def mode(self, df):
        list_name_of_attributes = df.columns;
        for i in list_name_of_attributes:
            #print(df[i].value_counts().index);
            if( df[i].isnull().sum()>0 and df[i].dtypes != 'object'):
                df[i].fillna(df[i].mode()[0], inplace=True);
            else:
                if (df[i].isnull().sum()>0 and df[i].dtypes == 'object'):
                    #print(df[i].value_counts().index);
                    df[i].fillna(df[i].value_counts().index.sort_values()[0], inplace=True);
        return;
    #---------------------------------------------------------------
    def median(self, df):
        list_name_of_attributes = df.columns;
        for i in list_name_of_attributes:
            #print(df[i].value_counts().index);
            if( df[i].isnull().sum()>0 and df[i].dtypes != 'object'):
                df[i].fillna(df[i].median(), inplace=True);
            else:
                if (df[i].isnull().sum()>0 and df[i].dtypes == 'object'):
                    #print(df[i].value_counts().index);
                    df[i].fillna(df[i].value_counts().index.sort_values()[0], inplace=True);
        return;
    #---------------------------------------------------------------
    def knn(self, df):
        return 0;

