"""
This module contains classes to work with database
use code with some adaptation from https://thinkdiff.net/how-to-use-python-sqlite3-using-sqlalchemy-158f9c54eb32
"""
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer,  MetaData, Float
from UserDefinedExceptions import NotAppropriateDatabaseTypeError
import sys

# Variables
SQLITE = 'sqlite'
# Table Names
TRAIN = 'training_functions'
IDEAL = 'ideal_functions'



class MyDatabase:
    """
    This class for creating object to work with database
    """
    DB_ENGINE = {
        SQLITE: 'sqlite:///{DB}'
    }
    # Main DB Connection Ref Obj
    db_engine = None

    def __init__(self, dbtype, username='', password='', dbname=''):
        """
        function to initialize an object and check and set parameters
        :param dbtype: type of the database. example - sqlite
        :param username: username to get access to the DB
        :param password: password to get access to the DB
        :param dbname: the name of DB
        """
        try:
            dbtype = dbtype.lower()
            if dbtype in self.DB_ENGINE.keys():
                engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
                self.db_engine = create_engine(engine_url)
                print(self.db_engine)
            else:
                raise NotAppropriateDatabaseTypeError(allowed_dbtypes=self.DB_ENGINE.keys())
        except NotAppropriateDatabaseTypeError:
            print(sys.exc_info()[1])
            sys.exit(1)
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            print("Error. Please check database parameters")
            print(''.join('[Error Message]: ' + str(exception_value) + ' ' +
                          '[Error Type]: ' + str(exception_type)))
            sys.exit(1)

    # create tables
    def create_db_tables(self):
        """
        function to create tables in the database
        """
        metadata = MetaData()

        train = Table(TRAIN, metadata,
                      Column('X', Float),
                      Column('Y1 (training function)', Float),
                      Column('Y2 (training function)', Float),
                      Column('Y3 (training function)', Float),
                      Column('Y4 (training function)', Float)
                      )

        ideal = Table(IDEAL, metadata,
                      Column('X', Float),
                      Column('Y1 (ideal function)', Float), Column('Y2 (ideal function)', Float),
                      Column('Y3 (ideal function)', Float), Column('Y4 (ideal function)', Float),
                      Column('Y5 (ideal function)', Float), Column('Y6 (ideal function)', Float),
                      Column('Y7 (ideal function)', Float), Column('Y8 (ideal function)', Float),
                      Column('Y9 (ideal function)', Float), Column('Y10 (ideal function)', Float),
                      Column('Y11 (ideal function)', Float), Column('Y12 (ideal function)', Float),
                      Column('Y13 (ideal function)', Float), Column('Y14 (ideal function)', Float),
                      Column('Y15 (ideal function)', Float), Column('Y16 (ideal function)', Float),
                      Column('Y17 (ideal function)', Float), Column('Y18 (ideal function)', Float),
                      Column('Y19 (ideal function)', Float), Column('Y20 (ideal function)', Float),
                      Column('Y21 (ideal function)', Float), Column('Y22 (ideal function)', Float),
                      Column('Y23 (ideal function)', Float), Column('Y24 (ideal function)', Float),
                      Column('Y25 (ideal function)', Float), Column('Y26 (ideal function)', Float),
                      Column('Y27 (ideal function)', Float), Column('Y28 (ideal function)', Float),
                      Column('Y29 (ideal function)', Float), Column('Y30 (ideal function)', Float),
                      Column('Y31 (ideal function)', Float), Column('Y32 (ideal function)', Float),
                      Column('Y33 (ideal function)', Float), Column('Y34 (ideal function)', Float),
                      Column('Y35 (ideal function)', Float), Column('Y36 (ideal function)', Float),
                      Column('Y37 (ideal function)', Float), Column('Y38 (ideal function)', Float),
                      Column('Y39 (ideal function)', Float), Column('Y40 (ideal function)', Float),
                      Column('Y41 (ideal function)', Float), Column('Y42 (ideal function)', Float),
                      Column('Y43 (ideal function)', Float), Column('Y44 (ideal function)', Float),
                      Column('Y45 (ideal function)', Float), Column('Y46 (ideal function)', Float),
                      Column('Y47 (ideal function)', Float), Column('Y48 (ideal function)', Float),
                      Column('Y49 (ideal function)', Float), Column('Y50 (ideal function)', Float)
                      )


        try:
            metadata.create_all(self.db_engine)
            # print("Tables are created")
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            print("Error occurred during table creation! Please check database parameters")
            print(''.join('[Error Message]: ' + str(exception_value) + ' ' +
                          '[Error Type]: ' + str(exception_type)))
            sys.exit(1)

    # Insert, Update, Delete with sql query
    def execute_query(self, query=''):
        """
        function to Insert, Update, Delete with sql query
        :param query: sql query to execute
        """
        if query == '':
            return
        with self.db_engine.connect() as connection:
            try:
                connection.execute(query)
            except Exception:
                exception_type, exception_value, exception_traceback = sys.exc_info()
                print("Error occurred during sql query execution! Please check the correctness of your sql query")
                print(''.join('[Error Message]: ' + str(exception_value) + ' ' +
                              '[Error Type]: ' + str(exception_type)))
                sys.exit(1)

    # Insert pandas DataFrame
    def insert_dataframe(self, df, table):
        """
        function to Insert pandas DF into the table
        :param df: pandas dataframe to insert
        :param table: the name of table to insert into
        """
        try:
            df.to_sql(table, con=self.db_engine, if_exists='replace', index=False)
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            print("Error occurred during sql query execution! Please check the correctness of pandas DF and the \
            name of database table")
            print(''.join('[Error Message]: ' + str(exception_value) + ' ' +
                          '[Error Type]: ' + str(exception_type)))
            sys.exit(1)

    # method will print all the data from a database table we provided as a parameter.
    def print_all_data(self, table='', query=''):
        """
        function to print data from the DB table
        :param query: sql query to execute
        :param table: the name of table to read dats from
        """
        query = query if query != '' else "SELECT * FROM '{}';".format(table)
        print(query)
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception:
                exception_type, exception_value, exception_traceback = sys.exc_info()
                print("Error occurred during sql query execution! Please check the correctness of your sql query")
                print(''.join('[Error Message]: ' + str(exception_value) + ' ' +
                              '[Error Type]: ' + str(exception_type)))
                sys.exit(1)
            # if there is no exception
            else:
                for row in result:
                    print(row)
                result.close()
        print("\n")

