
from keras import backend as K
# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)
# def mean_absolute_error(y_true, y_pred):
#     return K.mean(K.abs(y_pred - y_true), axis=-1)

def myloss(y_true,y_pred,MAE_loss_weight=0.6,MSE_loss_weight=0.6):
    '''
    自定义损失函数
    '''
    from keras import backend as K
    MAE_loss=K.mean(K.abs(y_pred-y_true),axis=-1)
    MSE_loss=K.mean(K.square(y_pred-y_true),axis=-1)
    
    total_loss=MAE_loss*MAE_loss_weight+MSE_loss*MSE_loss_weight
    return total_loss
