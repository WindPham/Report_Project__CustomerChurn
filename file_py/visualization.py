import matplotlib.pyplot as plt
import pandas as pd

class PLOT(object):
    #--------------------------------------------------------------------------------------------
    def __init__(self, name_file_csv):
        self.__df = pd.read_csv(name_file_csv);
        return;
    #--------------------------------------------------------------------------------------------
    def print_head(self):
        print(self.__df.head());
        return;
    #--------------------------------------------------------------------------------------------
    def print_tail(self):
        print(self.__df.tail());
        return;
    #--------------------------------------------------------------------------------------------
    def plt_line_two_columns(self, col_1_name, col_2_name):
        if self.__df[col_1_name].dtypes != 'object' and self.__df[col_2_name].dtypes != 'object':
            df = self.__df.sort_values(by = col_1_name)
            df.plot(x=col_1_name, y = col_2_name, figsize=(10,5), grid = True);
            plt.show();
        else :
            print("Can't draw!");
        return;
    #--------------------------------------------------------------------------------------------
    def plt_points_two_column(self, col_1_name, col_2_name):
        if self.__df[col_1_name].dtypes != 'object' and self.__df[col_2_name].dtypes != 'object':
            df = self.__df.sort_values(by = col_1_name)
            df.plot.scatter(x=col_1_name, y = col_2_name, figsize=(15,5), grid = True);
            plt.show();
        else :
            print("Can't draw!");
        return;
    #--------------------------------------------------------------------------------------------
    def plt_area_two_column(self, col_1_name, col_2_name):
        if(self.__df[col_2_name].dtypes == 'object'):
            if (scale == 'no'):
                df = self.__df[[col_1_name, col_2_name]];
                df = pd.get_dummies(df);
                df = df.groupby(col_1_name).count();
                df.plot(kind = 'area');
            plt.show();
        else:
            print("Can't draw!");
        return;
    #---------------------------------------------------------------------------------------------
    def plt_hist(self, col_name, Bins = 10):
        self.__df[col_name].plot(kind = 'hist', bins = Bins);
        plt.show();
        return;
    #---------------------------------------------------------------------------------------------
    def plt_bar(self, col_1_name, col_2_name):
        df =  self.__df[[col_1_name, col_2_name]];
        df = pd.get_dummies(df);
        df = df.groupby(col_1_name).count();
        print(df.head())
        df.plot(kind = 'bar');
        plt.show();
        return;




if __name__ == "__main__":
    df = PLOT("WA_Fn-UseC_-Telco-Customer-Churn-standard.csv");
    #df.plt_line_two_columns("tenure", "TotalCharges");
    #df.plt_points_two_column("tenure", "TotalCharges");
    #df.plt_points_two_column("tenure", "MonthlyCharges");
    #df.plt_points_two_column("MonthlyCharges", "TotalCharges");
    #df.plt_area_two_column("tenure", "TechSupport");
    #df.plt_area_two_column(col_1_name = "tenure", col_2_name = "gender");
    #df.plt_hist('tenure', 3);
    df.plt_bar('tenure','TechSupport');
	