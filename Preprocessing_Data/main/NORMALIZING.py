class NORMALIZING(object):
    """description of class"""
    def __init__(self):
        return;
    def MaxMin(self, one_column):
        if(one_column.dtypes != 'object'):
            one_column = ( one_column - one_column.min() ) / float( one_column.max() - one_column.min() );
        return;
    def zScores(self, one_column):
        if(one_column.dtypes != 'object'):
            mean = one_column.mean();
            std = one_column.std();
            one_column = (one_column-mean)/float(std);
        return;
    def MaxMin_for_all_columns(self, df):
        for i in df.columns:
            self.MaxMin(df[i]);
        return;
    def zScores_for_all_columns(self, df):
        for i in df.columns:
            self.zScores(df[i]);
        return;
