'''
Import the libraries

'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# list of classes to read train, test and ideal csv files
class TrainData(object):
    def trainer(self):
        try:
            # reading the train dataset from the path
            csv_path = os.path.join("train.csv")
            return pd.read_csv(csv_path)
        except Exception as e :
            print (" Error ! Code : {c} , Message , {m}"\
            .format ( c = e . code , m =str( e ) ) )

class IdealData(object):
    def idealer(self):
        try:
            csv_path = os.path.join("ideal.csv")
            return pd.read_csv(csv_path)
        except Exception as e :
            print (" Error ! Code : {c} , Message , {m}"\
            . format ( c = e . code , m =str( e ) ) )

class TestData(object):
    def tester(self):
        try:
            csv_path = os.path.join("test.csv")
            return pd.read_csv(csv_path)
        except Exception as e :
            print (" Error ! Code : {c} , Message , {m}"\
            .format ( c = e . code , m =str( e ) ) )

    
class SQLError ( Exception ) :
     pass

class Plot:
    def __init__(self, A, B,N):
        self.A = A
        self.B = B
        self.N = N

    def Show(self):
        """
        function to plot custom scatter plot 
        :param x: x-axis data
        :param y: y-axis data
        :param label_name1: it's a string and we can give the label 
        """
        plt.figure(figsize=(10,5), dpi=80)
        plt.grid(True,linestyle='--')
        plt.plot(self.A, self.B)
        plt.title(self.N)
        plt.xlabel("x")
        plt.show()
        
# Function to write the dataframe to sql
def result_to_sql ( fin_dfs ) :
    """
     Writes the data to a local sqlite db using pandas to.sql () method
     If the file already exists , it will be replaced
     : param file_name : the name the db gets
     : param suffix : to comply to the assignment the headers
     require a specific suffix to the original column name
    """
    try:
        dbcon = sqlite3.connect ("resultDB")
        curso = dbcon. cursor ()
    except Exception as err :
        raise SQLError (err)
    copy_of_function_data = fin_dfs.copy ()
    copy_of_function_data . to_sql (" final_result_df ",\
    con = dbcon ,if_exists =" replace ",index = True)
    check_res_df = pd.\
    read_sql_query(" select * from final_result_df limit 15", con = dbcon)
    print (check_res_df)
    curso.close ()


# Linear Regression function
def lin_reg(x,y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(y_train)
    return y_pred

def pca(x,y):
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn import utils
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2)
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y3=y.reshape(400,1)
    x1=x.reshape(400,1)
    y_transformed = lab.fit_transform(y3)

    # Fitting Logistic Regression To the training set
    from sklearn.linear_model import LogisticRegression 
    X_train=X_train.reshape(320,1)
    y_train=y_train.reshape(320,1)
    classifier = LogisticRegression()
    classifier.fit(x1, y_transformed)
    y_pred = classifier.predict(y_train)
    return y_pred

def svm(x,y):
    from sklearn import preprocessing
    from sklearn import utils
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2)
    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y2= y.reshape(400,1)
    x1 =x.reshape(400,1)
    y_transformed = lab.fit_transform(y2)
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state = 1)
    X_train=X_train.reshape(320,1)
    y_train=y_train.reshape(320,1)
    classifier.fit(x1,y_transformed)
    y_pred = classifier.predict(y_train)
    return y_pred

def arima(x):
    from pmdarima.arima import auto_arima
    from pmdarima.arima import ADFTest
    adf_test = ADFTest(alpha=0.05)
    x1=x.reshape(400,1)
    arima_model=auto_arima(x1,start_p=0,d=1,start_q=0,max_p=5,max_d=5,max_q=5,\
    start_P=0,D=1,start_Q=0,max_P=5,max_D=5,max_Q=5,m=12,seasonal=True,\
    error_action='warn',trace='true',suppress_warnings=True,\
    stepwise=True,random_state=12,n_fits=50)
    arima_model.summary()
    y_pred=pd.DataFrame(arima_model.predict(n_periods=20))
    y_pred.columns=["y_prediction"]
    return y_pred

def map_func(x):
    import math
    X = x.tolist()
    xy = map(math.sqrt, X)
    return X

def mse(df):
    y1=mean_squared_error(df["x"],df["y1"])
    y2=mean_squared_error(df["x"],df["y2"])
    y3=mean_squared_error(df["x"],df["y3"])
    y4=mean_squared_error(df["x"],df["y4"])
    y5=mean_squared_error(df["x"],df["y5"])
    y6=mean_squared_error(df["x"],df["y6"])
    y7=mean_squared_error(df["x"],df["y7"])
    y8=mean_squared_error(df["x"],df["y8"])
    y9=mean_squared_error(df["x"],df["y9"])
    y10=mean_squared_error(df["x"],df["y10"])
    lst=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
    lst.sort()
    lst=lst[0:4]
    return lst
