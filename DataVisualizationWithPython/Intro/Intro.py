import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
    def plt_points_with_labels_two_cols_and_label(self, col_1_name, col_2_name, label_name):
        if self.__df[col_1_name].dtypes != 'object' and self.__df[col_2_name].dtypes != 'object':
            df = self.__df.sort_values(by = col_1_name);
            labels = df[label_name].unique();
            markers = ['o', '^', '+', '-', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'];
            for i in range(0, len(labels)):
                plt.scatter(x=df[col_1_name], y = df[col_2_name], s = df[label_name]==labels[i], marker = markers[i]);
            plt.show();
        else :
            print("Can't draw!");
        return;
    #-------------------------------------------------------------------------------------------
    def plt_3d_scatter_three_cols(self, col_x, col_y, col_z):
        if self.__df[col_x].dtypes != 'object' and self.__df[col_y].dtypes != 'object' and self.__df[col_z].dtypes != 'object':
            df = self.__df[[col_x, col_y, col_z]];
            fig = plt.figure();
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[col_x], df[col_y], df[col_z], 'red');
            plt.show();
        return;
    #--------------------------------------------------------------------------------------------
    def plt_3d_scatter_three_cols_and_label(self, col_x, col_y, col_z, label_name):
        if self.__df[col_x].dtypes != 'object' and self.__df[col_y].dtypes != 'object' and self.__df[col_z].dtypes != 'object':
            df = self.__df[[col_x, col_y, col_z, label_name]];
            labels = df[label_name].unique();
            markers = ['o', '^', '+', '-', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'];
            fig = plt.figure();
            ax = fig.add_subplot(111, projection='3d')
            for i in range(0, len(labels)):
                ax.scatter(df[col_x], df[col_y], df[col_z], s = df[label_name]==labels[i], marker = markers[i]);
            plt.show();
        return;
    #---------------------------------------------------------------------------------------------
    def plt_3d_line_three_cols(self, col_x, col_y, col_z):
        if self.__df[col_x].dtypes != 'object' and self.__df[col_y].dtypes != 'object':
            df = self.__df[[col_x, col_y, col_z]];
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(df[col_x], df[col_y], df[col_z], cmap=plt.cm.viridis, linewidth=0.2)
            plt.show();
        else :
            print("Can't draw!");
        return;
    #----------------------------------------------------------------------------------------------
    def plt_scatter_two_cols_with_label_and_radius(self, col_1_name, col_2_name, label_name):
        Cs = ['red', 'blue', 'orange'];
        if self.__df[col_1_name].dtypes != 'object' and self.__df[col_2_name].dtypes != 'object':
            plt.figure(figsize = (100, 100))
            df = self.__df[ [col_1_name, col_2_name, label_name] ];
            labels = df[label_name].unique();
            for j in range(0, len(labels)):
                plt.scatter(df[col_1_name], df[col_2_name], 
                            s = 10*(df[col_2_name]/df[col_1_name])*(df[label_name]==labels[j]), 
                            c = Cs[j], 
                            cmap="Accent", alpha=0.6, edgecolors="white", linewidth=2); 
            plt.show();
        return;
    #----------------------------------------------------------------------------------------------
    def plt_scatter_three_cols_with_label_and_radius(self, col_time_name, col_1_name, col_2_name, label_name):
        if self.__df[col_1_name].dtypes != 'object' and self.__df[col_2_name].dtypes != 'object':
            df = self.__df.sort_values(by = col_1_name)[[col_time_name, col_1_name, col_2_name, label_name]];
            labels = df[label_name].unique();
            fig, axes = plt.subplots(int(len(df[col_time_name].unique())/2) + 1, 2, figsize = (100, 200));
            Cs = ['red', 'blue', 'orange']
            #fig.figsize=(1000, 1000000);
            #print(type(axes))
            for i, ax in enumerate(axes.flatten()):
                if (i==len(df[col_time_name].unique())):break;
                df_tmp = df [df[col_time_name] == i+1];#start 1 to 72
                for j in range(0, len(labels)):
                    S = df_tmp[ df_tmp[label_name] == j];
                    ax.scatter(df_tmp[col_1_name], df_tmp[col_2_name], 
                               s = 10*(df_tmp[col_2_name]/df_tmp[col_1_name])*(df_tmp[label_name]==labels[j]),
                               c = Cs[j], 
                               cmap="Accent", alpha=0.6, edgecolors="white", linewidth=2);
                    #df_tmp[label_name] == labels[j]
            plt.show();
        else:
            print("Can't draw!");
        return;