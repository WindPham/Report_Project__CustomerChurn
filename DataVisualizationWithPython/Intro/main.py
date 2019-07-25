
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
    df.plt_pie_two_cols('TechSupport', 'tenure');
    #df.plt_box('tenure');
    #df.plt_box_two_cols('TechSupport', 'tenure');
    #df.plt_hist_line('MonthlyCharges', two_axis = True);
    #df.plt_hist_line('MonthlyCharges', two_axis = False);
    #df.plt_hist_for_multi_statement('tenure','TechSupport');