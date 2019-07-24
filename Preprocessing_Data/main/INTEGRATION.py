import numpy as np
import pandas as pd



class INTEGRATION(object):
    """description of class"""
    def add_cols(self, df, list_cols_added, fixed_value):

        df.reindex(columns = list_cols_added).fillna(fixed_value);

        return;

    def entity_idenfication_column_not_similar(self, df1, df2):

        if (len(df1.columns)!=len(df2.columns)):
            print("Two input datasets have the other dim, so you have to match them before idenficating entities");
            return None;
        else:
            cols_1 = np.sort(df1.columns.values);
            cols_2 = np.sort(df2.columns.values);
            trade1 = len(cols_1)*[0];
            trade2 = len(cols_2)*[0];

            for i in range(0,len(cols_1)):
                for j in range(0,len(cols_2)):
                    if (cols_1[i] == cols_2[i]):
                        trade1[i]=trade2[j]=1;


        return;