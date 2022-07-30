"""
This module contains classes to define users exceptions
"""

class NotAppropriateDatabaseTypeError(Exception):
    """
    error to show that the given variable is not dataframe
    """

    def __init__(self, dbtypes):
        self.dbtypes_allowed = dbtypes

    def __str__(self):
        return "ERROR!! because the database type you are passing \
        is not found in database engine. \
            Allowed lists of database types is {0}. So Please check" \
            .format(self.dbtypes_allowed)