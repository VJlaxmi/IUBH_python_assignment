from visualization import *
import numpy as np
import pandas as pd
import sqlite3
import sqlalchemy
import os

# directory path for the csv files
# The program will load the three files from here
DLMD_PATH = r"C:\Users\pc\Downloads\Python Programming\Data"

# list of classes to read train, test and ideal csv files
class TrainData(object):
    def trainer(self, dataset= DLMD_PATH):
        # reading the train dataset from the path
        csv_path = os.path.join(dataset, "train.csv")
        return pd.read_csv(csv_path)

class IdealData(TrainData):
    def idealer(self, dataset= DLMD_PATH):
        csv_path = os.path.join(dataset, "ideal.csv")
        return pd.read_csv(csv_path)
    
    
class TestData(TrainData):
    def tester(self, dataset= DLMD_PATH):
        csv_path = os.path.join(dataset, "test.csv")
        return pd.read_csv(csv_path)

class SQLError(Exception):
    pass


def preparing_workdf(ideal,train):
    try:
        # copying the train_df and renaming the columns as y1_train, y2_train, y3_train, y4_train for better naming convention
        train_df1 = train.copy()
        train_df1.rename(columns={'y1':'y1_train', 'y2':'y2_train','y3':'y3_train','y4':'y4_train'}, inplace = True)
        # merging the ideal_df and train_df1 dataframes for further computation
        work = pd.merge(ideal, train_df1, on='x', how='inner')
        work.drop('x',axis='columns', inplace=True)
        return work
    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c = e.code, m = str(e)))
        
        

def y_traindataset(df,a):
    try:
        df1 = pd.DataFrame()
        for c in df.columns:
            if c == "x" or c == "y1_train" or c == "y2_train" or c == "y3_train" or c == "y4_train" :
                continue
            df1[c] = df.apply(lambda x: (x[a] - x[c])**2, axis=1)
        return df1
            
    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c = e.code, m = str(e)))
        
        
def training_dataframes(df):
    try:
        for c in df.columns:
            df[c] = df[c].sum()
        value_index = df[:1].T.idxmin()
        return value_index
    
    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c = e.code, m = str(e)))
        
def get_fin_res(res_df):
    fin_df = res_df[["x_test","y_test","match_fun","delta"]]
    fin_df.rename(columns={'x_test':' X (test func) ', 'y_test':'Y (test func)','delta':'Delta Y (test func)','match_fun':'No. of ideal func'}, inplace = True)
    dict_columns_type = {'No. of ideal func': float,'Delta Y (test func)': float}
    fin_df = fin_df.astype(dict_columns_type)
    return fin_df        
        
        
def testing_funct_for_ideal(test_df,train_df,ideal_df,y1_traindataset1,y2_traindataset2,y3_traindataset3,y4_traindataset4):
    try:
            # y1_train, y2_train, y3_train, y4_train and x from train_df
            y1_train = train_df["y1"]
            y2_train = train_df["y2"]
            y3_train = train_df["y3"]
            y4_train = train_df["y4"]
            x = train_df["x"]
            # getting all ideal functions for the y1, y2, y3, y4
            y1_ideal = ideal_df[y1_traindataset1]
            y2_ideal = ideal_df[y2_traindataset2]
            y3_ideal = ideal_df[y3_traindataset3]
            y4_ideal = ideal_df[y4_traindataset4]
            
            print("***************************************************************************************")
            print("Visualize y9_ideal function")
            custom_scatter_plot(ideal_df["x"], y1_ideal,"y9")
            
            print("Visualize y18_ideal function")
            custom_scatter_plot(ideal_df["x"], y2_ideal,"y18")
            
            print("Visualize y21_ideal function")
            custom_scatter_plot(ideal_df["x"], y3_ideal,"y21")
            
            print("Visualize y11_ideal function")
            custom_scatter_plot(ideal_df["x"], y4_ideal,"y11")
            
            print("***************************************************************************************")
        
            df1 = test_df.copy()
            xtest = df1["x"]
            ytest = df1["y"]
            
            map_dev_df = pd.DataFrame(columns=["x_test", 
                                                         "y_test", 
                                                         "y1_delta", 
                                                         "y2_delta", 
                                                         "y3_delta" ,
                                                         "y4_delta", 
                                                         "y1_matched",
                                                         "y2_matched",
                                                         "y3_matched",
                                                         "y4_matched",
                                                         "match_fun"
                                                        ])
        
            for xt,yt in zip(xtest,ytest):
                y1_delta = np.abs(y1_ideal[x==xt].values[0]-yt) - np.sqrt(2) * np.abs(y1_train[x==xt].values[0]-y1_ideal[x==xt].values[0])
                y2_delta = np.abs(y2_ideal[x==xt].values[0]-yt) - np.sqrt(2) * np.abs(y2_train[x==xt].values[0]-y2_ideal[x==xt].values[0])
                y3_delta = np.abs(y3_ideal[x==xt].values[0]-yt) - np.sqrt(2) * np.abs(y3_train[x==xt].values[0]-y3_ideal[x==xt].values[0])
                y4_delta = np.abs(y4_ideal[x==xt].values[0]-yt) - np.sqrt(2) * np.abs(y4_train[x==xt].values[0]-y4_ideal[x==xt].values[0])
                y1_matched = (y1_delta <= 0)
                y2_matched = (y2_delta <= 0)
                y3_matched = (y3_delta <= 0)
                y4_matched = (y4_delta <= 0)
                delta = max([y1_delta,y2_delta,y3_delta,y4_delta]) 
                match_fun = sum([y1_matched,y2_matched,y3_matched,y4_matched])
                map_dev_df = map_dev_df.append({
                    "x_test":xt,
                    "y_test":yt,
                    "y1_delta":y1_delta,
                    "y2_delta":y2_delta,
                    "y3_delta":y3_delta,
                    "y4_delta":y4_delta,  
                    "delta":delta,
                    "y1_matched":y1_matched,
                    "y2_matched":y2_matched,
                    "y3_matched":y3_matched,
                    "y4_matched":y4_matched,
                    'match_fun':match_fun
             
                },ignore_index=True)
                
            Y1_final_x_points = map_dev_df[map_dev_df['y1_matched']==True]['x_test']
            Y1_final_y_points = map_dev_df[map_dev_df['y1_matched']==True]['y_test']
                    
            Y2_final_x_points = map_dev_df[map_dev_df['y2_matched']==True]['x_test']
            Y2_final_y_points = map_dev_df[map_dev_df['y2_matched']==True]['y_test']
                
            Y3_final_x_points = map_dev_df[map_dev_df['y3_matched']==True]['x_test']
            Y3_final_y_points = map_dev_df[map_dev_df['y3_matched']==True]['y_test']
                
            Y4_final_x_points = map_dev_df[map_dev_df['y4_matched']==True]['x_test']
            Y4_final_y_points = map_dev_df[map_dev_df['y4_matched']==True]['y_test']
            
            # visualizing the result ideal datapoints on train data
            
            custom_plot(x,y1_ideal,Y1_final_x_points, Y1_final_y_points,label_name="y9_ideal",n="3")
            custom_plot(x,y2_ideal,Y2_final_x_points, Y2_final_y_points,label_name="y18_ideal",n="6")
            custom_plot(x,y3_ideal,Y3_final_x_points, Y3_final_y_points,label_name="y21_ideal",n="6")
            custom_plot(x,y4_ideal,Y4_final_x_points, Y4_final_y_points,label_name="y11_ideal",n="9")
            
            #print(map_dev_df)

            return map_dev_df

    except Exception as e:
        print('Error! Code: {c}, Message, {m}'.format(c = e.code, m = str(e)))



def result_to_sql(fin_dfs):
    """
    Writes the data to a local sqlite db using pandas to.sql() method
    If the file already exists, it will be replaced
    :param file_name: the name the db gets
    :param suffix: to comply to the assignment the headers require a specific suffix to the original column name
    """
    
    try:
        dbcon = sqlite3.connect('resultDB') 
        curso = dbcon.cursor() 
    except Exception as err:
        raise SQLError(err)
    
    
    #Using SQLalchemy an "engine" is created. It handles the creation of the db for us if it is not existent
    #engine = create_engine('sqlite:///{}.db'.format(file_name), echo=False)
    
    # Instead of writing an own implementation and possibly create bugs,
    # I decided to use functionality from Pandas to write to an sql db.
    # It only needs the "engine" object from sqlalchemy
    #print(fin_dfs)
    copy_of_function_data = fin_dfs.copy()
    copy_of_function_data.to_sql(
        "final_result_df",
        con=dbcon,
        if_exists="replace",
        index=True,
    )
    
    check_res_df = pd.read_sql_query('select * from final_result_df limit 15', con=dbcon)
    print(check_res_df)
    curso.close()
    
        
        
        
        
        