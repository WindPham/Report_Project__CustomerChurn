import sklearn as skl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
class DISCRETISATION(object):
    """description of class"""
    def __init__(self):
        return;
    def decision_tree(self, df, trans_col_name, label_col_name, Test_size = 0.3, Max_depth = 3, must_not_discre = []):
        if (trans_col_name not in must_not_discre) or len(must_not_discre) == 0:
            tree_model = DecisionTreeClassifier(max_depth = Max_depth);
            tree_model.fit(df[trans_col_name].to_frame(), df[label_col_name].to_frame());
            new_col_name = trans_col_name + "_" + label_col_name + "_Proba";
            predict_proba = tree_model.predict_proba(df[trans_col_name].to_frame());
            df[trans_col_name] = predict_proba[:,1];
            df[trans_col_name] = df[trans_col_name].astype(str);
        return;
    def decision_tree_all_cols(self, df, label_col_name, Test_size = 0.3, Max_depth = 3, must_not_discre = []):
        for i in df.columns:
            if df[i].dtypes != 'object' and i != label_col_name:
                self.decision_tree(df, i, label_col_name, Test_size, Max_depth, must_not_discre);
        return;

