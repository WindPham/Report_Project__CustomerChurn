import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
            df = self.__df[[col_1_name, col_2_name]];
            df = pd.get_dummies(df);
            df = df.groupby(col_1_name).agg(lambda x: x.eq(1).sum());
            df.plot(kind = 'area');
            plt.show();
        else:
            print("Can't draw!");
        return;
    #--------------------------------------------------------------------------------------------
    def plt_hist(self, col_name, Bins = 10, Normed = False, Edgecolor = 'black'):
        self.__df[col_name].plot(kind = 'hist', bins = Bins, normed = Normed, edgecolor = Edgecolor);
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_bar(self, col_1_name, col_2_name):
        df =  self.__df[[col_1_name, col_2_name]];
        df = pd.get_dummies(df);
        df = df.groupby(col_1_name).agg(lambda x: x.eq(1).sum());
        df.plot(kind = 'bar');
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_pie(self, col_1_name, col_2_name):
        df = self.__df.groupby(col_1_name).sum();
        df[col_2_name].plot(kind = 'pie');
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_pie_two_cols(self, col_1_name, col_2_name):
        df = self.__df[[col_1_name,col_2_name]];
        df = pd.get_dummies(df);
        df = df.groupby(col_2_name).agg(lambda x: x.eq(1).sum());
        #label = df[col_2_name].to_frame();
        print(df.sum(axis = 1));
        #r_mean = np.mean(df.sum(axis = 1));
        label = df.index;
        df = df.transpose();
        
        fig, axes = plt.subplots(int(len(label)/12), 12);
        for i, ax in enumerate(axes.flatten()):
            if (i==len(label)):break;
            ax.pie(df[i+1], radius = 1.3, autopct="%.1f%%", pctdistance=0.5)
            ax.set_title(str(i+1) + "_" + str(df[i+1].sum()) );
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_box(self, col_name):
        self.__df[col_name].plot(kind = 'box');
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_box_two_cols(self, col_1_name, col_2_name):
        df = self.__df[[col_1_name,col_2_name]];
        df = pd.get_dummies(df);
        df = df.groupby(col_2_name).agg(lambda x: x.eq(1).sum());
        df.plot(kind = 'box');
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_hist_line(self, col_name, Bins = 20, Edgecolor = 'black', two_axis = False):
        if two_axis == False:
            self.__df[col_name].plot(kind = 'hist', bins = Bins, normed = True, edgecolor = Edgecolor);
            self.__df[col_name].plot(kind = 'kde')
            plt.show();
        else:
            fig,ax = plt.subplots();
            self.__df[col_name].plot(kind = 'hist', bins = Bins, normed = True, edgecolor = Edgecolor);
            ax_other = ax.twinx();
            self.__df[col_name].plot(kind = 'kde')
            plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_hist_for_multi_statement(self, col_1_name, col_2_name):
        df =  self.__df[[col_1_name, col_2_name]];
        df = pd.get_dummies(df);
        df = df.groupby(col_1_name).agg(lambda x: x.eq(1).sum());
        #df = df.transpose();
        df.plot(kind = 'hist', bins = 50);
        plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    