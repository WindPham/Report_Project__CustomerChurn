
import Intro 

if __name__ == "__main__":
    df = Intro.PLOT("WA_Fn-UseC_-Telco-Customer-Churn-standard.csv");
    #df.plt_line_two_columns("tenure", "TotalCharges");
    #df.plt_points_two_column("tenure", "TotalCharges");
    #df.plt_points_two_column("tenure", "MonthlyCharges");
    #df.plt_points_two_column("MonthlyCharges", "TotalCharges");
    #df.plt_area_two_column("tenure", "TechSupport");
    #df.plt_area_two_column(col_1_name = "tenure", col_2_name = "gender");
    #df.plt_hist('tenure', 3);
    #df.plt_bar('tenure','gender')
    #df.plt_pie('TechSupport', 'tenure');
    #df.plt_pie_two_cols('TechSupport', 'tenure');
    #df.plt_box('tenure');
    #df.plt_box_two_cols('TechSupport', 'tenure');
    #df.plt_hist_line('MonthlyCharges', two_axis = True);
    #df.plt_hist_line('MonthlyCharges', two_axis = False);
    #df.plt_hist_for_multi_statement('tenure','TechSupport');
    #df.plt_points_with_labels_two_cols_and_label("tenure", "TotalCharges", 'TechSupport');
    #df.plt_3d_scatter_three_cols('tenure', 'MonthlyCharges', 'TotalCharges');
    #df.plt_3d_scatter_three_cols_and_label('tenure', 'MonthlyCharges', 'TotalCharges', 'Churn');
    #df.plt_3d_line_three_cols('tenure', 'MonthlyCharges', 'TotalCharges');
    #df.plt_scatter_two_cols_with_label_and_radius('MonthlyCharges', "TotalCharges", 'Churn');
    df.plt_scatter_three_cols_with_label_and_radius('tenure', 'MonthlyCharges', "TotalCharges", 'Churn');
