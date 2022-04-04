"""
This module contains classes to define users exceptions
"""


class DoNotHaveColumnError(Exception):
    """
    error to show that dataset doesn't have some specific column name doesn't
    """

    def __init__(self, column_name, file_source):
        self.column_name = column_name
        self.file_source = file_source

    def __str__(self):
        return "Error!!! Data from {0} doesn't have column name {1}. Please check your source files".format(
            self.file_source,
            self.column_name)


class EmptyDataError(Exception):
    """
    error to show that dataset is empty
    """

    def __init__(self, file_source):
        self.file_source = file_source

    def __str__(self):
        return "Error!!! Data from {0} is empty. Please check your source files".format(self.file_source)


class NotAppropriateLengthError(Exception):
    """
    error to show that the len of given object is wrong
    """

    def __init__(self, obj, obj_len, len_appropriate):
        self.obj = obj
        self.obj_len = obj_len
        self.len_appropriate = len_appropriate

    def __str__(self):
        return "Error!!! The raw in given object {0} is inappropriate and equal to {1}. It should be equal to {2}" \
            .format(self.obj, self.obj_len, self.len_appropriate)


class NotAppropriateColumnError(Exception):
    """
    error to show that the given dataframes do not have column col_name
    """

    def __init__(self, col_name, dataframes):
        self.col_name = col_name
        self.dataframes = dataframes

    def __str__(self):
        return "Error!!! The  given dataframes {0} and {1} do not have column {2}. Please check your source files" \
            .format(self.dataframes[0].head(3), self.dataframes[1].head(3), self.col_name)


class NotAppropriateDataFormatError(Exception):
    """
    error to show that in the given dataframe all values are not numbers
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __str__(self):
        return "Error!!! In the given dataframe {0} all values are not numbers. Please check your source files" \
            .format(self.dataframe.head(3))


class NotAppropriateVariableFormatError(Exception):
    """
    error to show that the given variable is not dataframe
    """

    def __init__(self, var):
        self.var = var

    def __str__(self):
        return "Error!!! In the given variable {0} is not dataframe. Please check" \
            .format(self.var)


class NotAppropriateDatabaseTypeError(Exception):
    """
    error to show that the given variable is not dataframe
    """

    def __init__(self, allowed_dbtypes):
        self.allowed_dbtypes = allowed_dbtypes

    def __str__(self):
        return "Error!!! DBType is not found in DB_ENGINE. Allowed lists of DBTypes is {0}. Please check" \
            .format(self.allowed_dbtypes)