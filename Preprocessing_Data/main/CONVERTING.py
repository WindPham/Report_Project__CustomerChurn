import AllLibrary

class CONVERTING(object):
    """description of class"""
    def __init__(self):
        return;

    def toNumeric(self, one_column):
        # <class 'pandas.core.series.Series'>
        isFloat = True;
        if len(one_column)>0:
            try:
                float(one_column[0]);
                isFloat = True;
            except ValueError:
                isFloat = False;

        if isFloat == True:
            one_column = pd.to_numeric(one_column, errors='coerce');
        return isFloat;

    def toString(self, one_column):
        #name_of_column = one_column.name;
        one_column = one_column.astype(str);
        return True;

    def label(self,df, label_col_name):
        df[label_col_name].replace(to_replace='Yes', value=1, inplace=True);
        df[label_col_name].replace(to_replace='No',  value=0, inplace=True);
        return;

