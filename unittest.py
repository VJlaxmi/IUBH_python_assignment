import unittest
from function import training_dataframes

# importing pandas as pd
import pandas as pd

class ProjectcsvTest(unittest.TestCase):
 
 def test_func(self):
     '''
     unit testing in 3steps
     a) create an object
     b) create the computation using the function to be tested
     c) use the assert function
     '''
     # Creating the dataframe 
     df = pd.DataFrame({"A":[4, 5, 2, 6],
                   "B":[11, 2, 5, 8],
                   "C":[1, 8, 66, 4]})
     result = training_dataframes(df)
     self.assertAlmostEqual(result, "A")
  
 
if __name__ == '__main__':
    unittest.main
 