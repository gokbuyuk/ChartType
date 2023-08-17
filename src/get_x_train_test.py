import numpy as np
import pandas as pd
import os
from icecream import ic
from matplotlib import pyplot as plt

def get_x_train_test(x_all, y_all, y_train, y_test, id_column='filename'):
    """
    Extracts and returns the training and testing data from the numpy array of images and labels.

    Args:
        x_all (numpy.ndarray): Numpy array of all images
        y_all (pandas.DataFrame): DataFrame of all labels
        y_train (pandas.DataFrame): DataFrame of training labels
        y_test (pandas.DataFrame): DataFrame of testing labels
        id_column (str, optional): Name of the column in y_all, y_train and y_test that contains the identifier of files. /
            Defaults to 'filename'.

    Returns:
        dict: Dictionary with the following keys:
              - 'x_train': Numpy array of training images
              - 'x_test': Numpy array of testing images
    """
    # get the number of rows in the dataframes
    num_rows_train = y_train.shape[0]
    num_rows_test = y_test.shape[0]


    # create empty numpy arrays
    img_height = x_all.shape[1]
    img_width = x_all.shape[2]
    num_channels = x_all.shape[3]
    x_train = np.empty((num_rows_train, img_height, img_width, num_channels), dtype=np.float32)
    x_test = np.empty((num_rows_test, img_height, img_width, num_channels), dtype=np.float32)

    # loop through each row in y_train dataframe and extract arrays from .npy files
    for i, filename in enumerate(y_train[id_column]):
        index = y_all.index[y_all[id_column] == filename].tolist()[0]
        x_train[i] = x_all[index]

    # loop through each row in y_test dataframe and extract arrays from .npy files
    for i, filename in enumerate(y_test[id_column]):
        index = y_all.index[y_all[id_column] == filename].tolist()[0]
        x_test[i] = x_all[index]
        
    ic(x_train.shape, x_test.shape)
    
    return({'x_train': x_train,
            'x_test': x_test})
    
    
    
    
if __name__ == '__main__':
    
    
    
    
    
    
    
    
    #########################################################
    ###### This part is to use the get_x_train_test function on kaggle training data
    # data_dir = os.path.join('data','data_ml','processed')
    # xaxis_all = np.load(os.path.join(data_dir,'xaxis_all.npy'))
    # yaxis_all = np.load(os.path.join(data_dir,'yaxis_all.npy'))

    # tick_types_all = pd.read_csv(os.path.join(data_dir,'tick_types_all.csv'), header=0)

    # y_train_xtick = pd.read_csv(os.path.join(data_dir,'y_train_xtick_type.csv'), header=0)
    # y_test_xtick = pd.read_csv(os.path.join(data_dir,'y_test_xtick_type.csv'), header=0)

    # y_train_ytick = pd.read_csv(os.path.join(data_dir,'y_train_ytick_type.csv'), header=0)
    # y_test_ytick = pd.read_csv(os.path.join(data_dir,'y_test_ytick_type.csv'), header=0)

    # ic(xaxis_all.shape, yaxis_all.shape, tick_types_all.shape, 
    # y_train_xtick.shape, y_test_xtick.shape,
    # y_train_ytick.shape, y_test_ytick.shape)
    
    # x_train_test_xtick = get_x_train_test(y_all = tick_types_all,
    #                                    x_all = xaxis_all, 
    #                                    y_train = y_train_xtick,
    #                                    y_test = y_test_xtick
    #                                    )
    # x_train_xtick = x_train_test_xtick['x_train']
    # x_test_xtick = x_train_test_xtick['x_test']
    
    # x_train_test_ytick = get_x_train_test(y_all = tick_types_all,
    #                                    x_all = yaxis_all, 
    #                                    y_train = y_train_ytick,
    #                                    y_test = y_test_ytick
    #                                     )
    # x_train_ytick = x_train_test_ytick['x_train']
    # x_test_ytick = x_train_test_ytick['x_test']
    
    # for arrname in ['x_train_xtick', 'x_test_xtick', 'x_train_ytick', 'x_test_ytick']:
    #     ic(arrname)
    #     arr = locals()[arrname]
    #     ic(arr.shape)
    #     np.save(os.path.join(data_dir, f'{arrname}.npy'), arr)
    #     for i in range(arr.shape[0]):
    #         if np.unique(arr[i]).size == 1:
    #             warning_msg = f"Array at index {i} in {arrname} contains all same values."
    #             raise AssertionError(warning_msg)


