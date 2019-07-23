class INTEGRATION(object):
    """description of class"""
    def add_cols(self, df, list_cols_added, fixed_value):

        df.reindex(columns = list_cols_added).fillna(fixed_value);

        return;

