import math
import numpy as np
import pandas as pd
import os
from core.utils import get_file
import matplotlib.pyplot as plt

def func(path):
    temp = []
    for i in os.listdir(path):
        path2 = os.path.join(path,i)  #拼接绝对路径
        if os.path.isdir(path2):      #判断如果是文件夹,调用本身
            func(path2)
        else:
            temp.append(i)
    return temp
class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.split = split
        self.cols = cols


    def get_train_datas(self, seq_len, normalise,train_path):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for file in get_file(train_path) :
            # if 'D9' in file:
            filename = os.path.join(train_path, file)
            dataframe = pd.read_csv(filename)
            i_split = int(len(dataframe) * self.split)
            data_train = dataframe.get(self.cols).values[:i_split]
            len_train  = len(data_train)
                
            for i in range(len_train - seq_len):
                x, y = self._next_windows(i, seq_len, normalise,data_train)
                data_x.append(x)
                data_y.append(y)
        return np.array(data_x), np.array(data_y)
    def _next_windows(self, i, seq_len, normalise,data_train):
        '''Generates the next data window from the given index location i'''
        window = data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []                   
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        
        return np.array(data_x), np.array(data_y)
    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''

        # for file in self.files :
        #     filename = os.path.join('data\csv', file)
        #     dataframe = pd.read_csv(filename)
        #     i_split = int(len(dataframe) * self.split)
        #     data_train = dataframe.get(self.cols).values[:i_split]
        #     len_train  = len(data_train)
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)
    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
   


    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            
            for col_i in range(window.shape[1]):
                normalised_col = [(float(p) / float(window[0, col_i]) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)