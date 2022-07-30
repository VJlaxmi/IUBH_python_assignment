# Module to work with visulizations
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

# module to work with sqlite database
import sqlite3
import sqlalchemy

from functions import *

from sklearn.decomposition import PCA


# SET OPTIONS to work comfortable with Pandas Dataframe
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)

if __name__ == "__main__":
    
    # Import train, test and ideal data from files
    train_df = TrainData().trainer()
    test_df = TestData().tester()
    ideal_df = IdealData().idealer()
    
    print("Printing the shape of data frames loaded")
    print("train dataframe shape is",train_df.shape)
    print("test dataframe shape is", test_df.shape)
    print("ideal dataframe shape is",ideal_df.shape)
    
    print("Visualize y1 function from train dataframe")
    Plot(train_df["x"],train_df["y1"],"y1")
    print("Visualize y2 function from train dataframe")
    Plot(train_df["x"],train_df["y2"],"y2")
    print("Visualize y3 function from train dataframe")
    Plot(train_df["x"],train_df["y3"],"y3")
    print("Visualize y4 function from train dataframe")
    Plot(train_df["x"],train_df["y4"],"y4")
    
    # Create Tables
    dbms = mydatabase . MyDatabase (mydatabase.SQLITE ,dbname =’mysqldb.sqlite’)
    dbms . create_db_tables ()
    # insert train data from pandas dataframe
    dbms . insert_dataframe ( df = train_df , table =’ training_functions ’)
    # insert ideal data from pandas dataframe
    dbms . insert_dataframe ( df = ideal_df , table =’ ideal_functions ’)
    
    # correlation matrix for the training dataset
    sns.heatmap(train_df.corr())
    
    # Reshaping the training dataset
    x = train_df["x"].values.reshape(200,2)
    y1 = train_df["y1"].values.reshape(200,2)
    y2 = train_df["y2"].values.reshape(200,2)
    y3 = train_df["y3"].values.reshape(200,2)
    y4 = train_df["y4"].values.reshape(200,2)
    
    # Linear Regression for x and y4
    lin_y_pred = lin_reg(x,y4)
    print("regression coeffs",regressor.coef_)
    print("regression intercepts",regressor.intercept_)
    
    # Fit PCA for x and y3
    pca_y_pred = pca(x,y3)
    
    # Fit support vector machine model for x and y2
    svm_y_pred = svm(x,y2)
    
    # arima model
    arima_y_pred = arima(x)
    
    #mse
    list_mse = mse(ideal_df)
    
    # Map function for test dataset
    map_val = map_func(test_df['x'])
    
    # Writing result to database
    result_to_sql(fin_df)
    
    

    
    
    
    
    
    






