import unittest
import warnings
import numpy as np
import pandas as pd
from icecream import ic
from get_x_train_test import get_x_train_test

class TestGetXTrainTest(unittest.TestCase):
    
    def test_get_x_train_test(self):
        # create some dummy data
        zero_array = np.zeros((128, 128, 3))
        
        x_all = np.random.rand(100, 128, 128, 3)
        x_all = np.append(x_all, [zero_array], axis=0)

        y_all = pd.DataFrame({'filename': ['img{}.jpg'.format(i) for i in range(100)] + ['zero_img']})
        y_train = y_all.iloc[:80]
        y_test = y_all.iloc[80:]
        
        # call the function to get x_train and x_test
        result = get_x_train_test(x_all, y_all, y_train, y_test)
        x_train = result['x_train']
        x_test = result['x_test']
        
        # check that the shapes are correct
        self.assertEqual(x_train.shape, (80, 128, 128, 3))
        self.assertEqual(x_test.shape, (21, 128, 128, 3))
        
        for arr_name in ['x_train', 'x_test']:
            ic(arr_name)
            arr = locals()[arr_name]
            ic(arr.shape)
            for i in range(arr.shape[0]):
                if np.unique(arr[i]).size == 1:
                    warning_msg = f"Array at index {i} in {arr_name} contains all same values."
                    raise AssertionError(warning_msg)

        
if __name__ == '__main__':
    unittest.main()
