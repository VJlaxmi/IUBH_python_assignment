'''
list of imports
'''

# Module to work with visulizations
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# module to work with sqlite database
import sqlite3
import sqlalchemy
import mydatabase

from function import *

# SET OPTIONS to work comfortable with Pandas Dataframe
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)


if __name__ == "__main__":
    
    # Import train, test and ideal data from files
    train_df = TrainData().trainer()
    test_df = TestData().tester()
    ideal_df = IdealData().idealer()
    
    print('*****************************************************************************************')
    
    print("Printing the shape of data frames loaded")
    print("train dataframe shape is",train_df.shape)
    print("test dataframe shape is", test_df.shape)
    print("ideal dataframe shape is",ideal_df.shape)
    
    print("****************************************************************************************")
    
    print("Visualize y1 function from train dataframe")
    custom_scatter_plot(train_df["x"],train_df["y1"],"y1")
    print("Visualize y2 function from train dataframe")
    custom_scatter_plot(train_df["x"],train_df["y2"],"y2")
    print("Visualize y3 function from train dataframe")
    custom_scatter_plot(train_df["x"],train_df["y3"],"y3")
    print("Visualize y4 function from train dataframe")
    custom_scatter_plot(train_df["x"],train_df["y4"],"y4")
    
    print("***************************************************************************************")
    
    # Create Tables
    dbms = mydatabase.MyDatabase(mydatabase.SQLITE, dbname='mysqldb.sqlite')
    dbms.create_db_tables()
    # insert train data from pandas dataframe
    dbms.insert_dataframe(df=train_df, table='training_functions')
    # insert ideal data from pandas dataframe
    dbms.insert_dataframe(df=ideal_df, table='ideal_functions')
    
    print("***************************************************************************************")
    
    # Find 4 ideal functions for given train data
    # preparing dataframe to apply least square method to find out minimum deviation points
    work_df = preparing_workdf(ideal_df,train_df)
    y1_traindataset = y_traindataset(work_df,a='y1_train')
    y2_traindataset = y_traindataset(work_df,a='y2_train')
    y3_traindataset = y_traindataset(work_df,a='y3_train')
    y4_traindataset = y_traindataset(work_df,a='y4_train')
    # getting the best ideal functions
    y1_traindataset1 = training_dataframes(y1_traindataset)
    y2_traindataset2 = training_dataframes(y2_traindataset)
    y3_traindataset3 = training_dataframes(y3_traindataset)
    y4_traindataset4 = training_dataframes(y4_traindataset)
    
    # getting the result dataframe
    res_df = testing_funct_for_ideal(test_df,train_df,ideal_df,y1_traindataset1,y2_traindataset2,y3_traindataset3,y4_traindataset4)
    
    #print(res_df)
    fin_df = get_fin_res(res_df)

    # writing results to sql
    result_to_sql(fin_df)    
    
    


