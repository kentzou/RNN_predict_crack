import datetime as dt
import os
from keras import backend as K

def get_file(path):
    temp = []
    for i in os.listdir(path):
        path2 = os.path.join(path, i) 
        if os.path.isdir(path2): 
            get_file(path2)
        else:
            temp.append(i)
    return temp
def mae_mse(y_true, y_pred):
    MAE_loss = K.mean(K.abs(y_pred-y_true), axis=-1)
    MSE_loss = K.mean(K.square(y_pred-y_true), axis=-1)
    return MAE_loss*0.1+MSE_loss*0.9
class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))