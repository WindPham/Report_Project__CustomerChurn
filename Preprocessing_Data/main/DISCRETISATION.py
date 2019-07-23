import sklearn as skl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from IPython.display import Image
from IPython.core.display import HTML 
import pydot
import pydotplus
import graphviz
from sklearn.externals.six import StringIO
import os
import numpy as np


class DISCRETISATION(object):
    """description of class"""
    def __init__(self):
        return;
    #===================================================================================================
    def decision_tree(self, df, trans_col_name, label_col_name, Max_depth = 3, must_not_discre = []):
        if (trans_col_name not in must_not_discre) or len(must_not_discre) == 0:
            tree_model = DecisionTreeClassifier(max_depth = Max_depth);
            tree_model.fit(df[trans_col_name].to_frame(), df[label_col_name].to_frame());
            new_col_name = trans_col_name + "_" + label_col_name + "_Proba";
            predict_proba = tree_model.predict_proba(df[trans_col_name].to_frame());
            df[trans_col_name] = predict_proba[:,1];
            df[trans_col_name] = df[trans_col_name].astype(str);
            return tree_model, trans_col_name;
        return None, trans_col_name;
    #===================================================================================================
    def decision_tree_all_cols(self, df, label_col_name, Max_depth = 3, must_not_discre = []):
        result = [];
        for i in df.columns:
            if df[i].dtypes != 'object' and i != label_col_name:
                model = self.decision_tree(df, i, label_col_name, Max_depth, must_not_discre);
                result.append(model);
        return result;
    #===================================================================================================
    def random_forest(self, df, trans_col_name, label_col_name, Max_depth = 3, must_not_discre = []):
        if (trans_col_name not in must_not_discre) or len(must_not_discre) == 0:
            tree_model = RandomForestClassifier(max_depth=Max_depth)
            tree_model.fit(df[trans_col_name].to_frame(), df[label_col_name].to_frame());
            new_col_name = trans_col_name + "_" + label_col_name + "_Proba";
            predict_proba = tree_model.predict_proba(df[trans_col_name].to_frame());
            df[trans_col_name] = predict_proba[:,1];
            df[trans_col_name] = df[trans_col_name].astype(str);
            return tree_model, trans_col_name;
        return None, trans_col_name;
    #===================================================================================================
    def random_forest_all_cols(self, df, label_col_name, Max_depth = 3, must_not_discre = []):
        result = [];
        for i in df.columns:
            if df[i].dtypes != 'object' and i != label_col_name:
                model = self.random_forest(df, i, label_col_name, Max_depth, must_not_discre);
                result.append(model);
        return result;
    #===================================================================================================
    def support_other_set(self, models, df, sample, must_not_discre = []):
        for i in range(len(models)):
            (model, col) = models[i];
            if(model != None):
                predict_proba = model.predict_proba(df[col].to_frame());
                df[col] = predict_proba[:,1];
                df[col] = df[col].astype(str);
        return;
