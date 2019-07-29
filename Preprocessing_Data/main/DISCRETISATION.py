import sklearn as skl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from IPython.display import Image
from IPython.core.display import HTML 
from sklearn.externals.six import StringIO
import pydot
import pydotplus
import graphviz
import os
import numpy as np


class DISCRETISATION(object):
    """description of class"""
    #======================================================================================================
    def __init__(self):


        return;
    #======================================================================================================
    def binning_frequency(self, df, col_name, Bins):
        df.reset_index(drop=True,inplace = True);
        df.sort_values(col_name);
        m = int(len(df[col_name])/Bins);
        for i in range(0,Bins):
            if(i==Bins-2):
                df.set_value(range(i*m, len(df[col_name])), col_name, str(i));
            else:
                df.set_value(range(i*m, i*m + m), col_name, str(i));
        return;
    #======================================================================================================
    def binning_width(self, df, col_name, Bins):
        df.reset_index(drop=True,inplace = True);
        Max = np.max(df[col_name]);
        Min = np.min(df[col_name]);
        W = int((Max - Min)/Bins);
        df[col_name] /= W;
        df[col_name]=df[col_name].astype(int);
        df.loc[df[col_name]==Bins, col_name]=Bins-1;
        df[col_name].astype(str);
        return;
    #======================================================================================================
    def k_mean(self, df, col_name, Bins):
        df.reset_index(drop=True,inplace = True);
        model = KMeans(n_clusters = Bins, random_state = 0);
        model.fit(df[col_name].values.reshape((-1,1)));
        label = model.labels_;
        df[col_name] = label;
        return;
    #======================================================================================================
    def decision_tree(self, df, trans_col_name, label_col_name, Max_depth = 3, must_not_discre = []):
        df.reset_index(drop=True,inplace = True);
        if (trans_col_name not in must_not_discre) or len(must_not_discre) == 0:
            tree_model = DecisionTreeClassifier(max_depth = Max_depth);
            tree_model.fit(df[trans_col_name].to_frame(), df[label_col_name].to_frame());
            new_col_name = trans_col_name + "_" + label_col_name + "_Proba";
            predict_proba = tree_model.predict_proba(df[trans_col_name].to_frame());
            df.loc[:,new_col_name] = predict_proba[:,1];
            min_max = pd.concat(
                [df.groupby([new_col_name])[trans_col_name].min(),
                 df.groupby([new_col_name])[trans_col_name].max()], 
                 axis=1);
            return min_max, trans_col_name;
        return None, trans_col_name;
    #======================================================================================================
    def decision_tree_all_cols(self, df, label_col_name, Max_depth = 3, must_not_discre = []):
        df.reset_index(drop=True,inplace = True);
        result = [];
        for i in df.columns:
            if df[i].dtypes != 'object' and i != label_col_name:
                result.append(decision_tree(df, i, label_col_name, Max_depth, must_not_discre));
        return result;
    #======================================================================================================
    def random_forest(self, df, trans_col_name, label_col_name, Max_depth = 3, must_not_discre = []):
        df.reset_index(drop=True,inplace = True);
        if (trans_col_name not in must_not_discre) or len(must_not_discre) == 0:
            tree_model = RandomForestClassifier(n_estimators=100, max_depth=Max_depth)
            tree_model.fit(df[trans_col_name].to_frame(), df[label_col_name].to_frame());
            new_col_name = trans_col_name + "_" + label_col_name + "_Proba";
            predict_proba = tree_model.predict_proba(df[trans_col_name].to_frame());
            df.loc[:, new_col_name] = predict_proba[:,1];
            min_max = pd.concat(
                [df.groupby([new_col_name])[trans_col_name].min(),
                 df.groupby([new_col_name])[trans_col_name].max()], 
                 axis=1);
            self.support(df, new_col_name, min_max);
            df[trans_col_name] = df[new_col_name];
            df.drop(new_col_name, axis = 1, inplace = True);
            return min_max, trans_col_name;
        return pd.DataFrame(), trans_col_name;
    #======================================================================================================
    def random_forest_all_cols(self, df, label_col_name, Max_depth = 3, must_not_discre = []):
        df.reset_index(drop=True,inplace = True);
        result = [];
        for i in df.columns:
            if df[i].dtypes != 'object' and i != label_col_name:
                intervals, col = self.random_forest(df, i, label_col_name, Max_depth = Max_depth, must_not_discre = must_not_discre)
                result.append((intervals, col));
        return result;
    #======================================================================================================
    def support(self, df, col_name, min_max_intervals):
        df.reset_index(drop=True,inplace = True);
        df.loc[:, 'min_'+col_name] = np.nan;
        df.loc[:, 'max_'+col_name] = np.nan;
        for i in range(0, df[col_name].shape[0]):
            for j in range(0, min_max_intervals.shape[0]):
                if df[col_name].values[i] <= min_max_intervals.values[j,1] and df[col_name].values[i] >= min_max_intervals.values[j,0]:
                    print("%s : %s"%(df.columns[i], df.columns[j]));
                    df.set_value(i,'min_'+col_name, min_max_intervals.values[j,0]);
                    df.set_value(i,'max_'+col_name, min_max_intervals.values[j,1]);
                    break;
        df['min_'+col_name] = df['min_'+col_name].astype(str);
        df['max_'+col_name] = df['max_'+col_name].astype(str);
        df[col_name] = df['min_'+col_name];
        df = df.drop(['min_'+col_name, 'max_'+col_name],axis = 1, inplace=True);
        return;
    #======================================================================================================
    def support_all(self, df, list_intervals):
        df.reset_index(drop=True,inplace = True);
        for i in list_intervals:
            intervals, col = i;
            print(intervals.empty);
            if (intervals.empty == True):
                self.support(df, col, intervals);
