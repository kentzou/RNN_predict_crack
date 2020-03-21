import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras import metrics
from core.utils import Timer
from keras import backend as K
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, Embedding, Bidirectional
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# 自定义损失函数
def mae_mse(y_true, y_pred):
    MAE_loss = K.mean(K.abs(y_pred-y_true), axis=-1)
    MSE_loss = K.mean(K.square(y_pred-y_true), axis=-1)

    return MAE_loss*0.1+MSE_loss*0.9



class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()
        self.file_list = []

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'gru':
                self.model.add(Bidirectional(GRU(neurons, input_shape=(
                    input_timesteps, input_dim), return_sequences=return_seq)))
                # self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'lstm':
                self.model.add(Bidirectional(LSTM(neurons, input_shape=(
                    input_timesteps, input_dim), return_sequences=return_seq)))
                # self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        #self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'],metrics=[configs['model']['acc']])
        self.model.compile(loss=mae_mse, optimizer=configs['model']['optimizer'], metrics=[
                           configs['model']['acc']])
        # mae

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, configs):
        timer = Timer()
        timer.start()
        #print('[Model] Training Started')
        #print('[Model] %s epochs, %s batch size' % (configs['training']['epochs'], configs['training']['batch_size']))
        model_file = os.path.join(
            '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['training']['epochs'])))
        save_fname = os.path.join(configs['model']['save_dir'], model_file)
        log_dir = os.path.join(configs['data']['log_dir'], '%s-e%s' % (
            dt.datetime.now().strftime('%d%m%Y-%H%M'), str(configs['training']['epochs'])))
        tensorboard = TensorBoard(log_dir=log_dir)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname,
                            monitor='val_loss', save_best_only=True),
            tensorboard
        ]
        history = self.model.fit(
            x,
            y,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            callbacks=callbacks
        )
        # save_fname = os.path.join('%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['training']['epochs'])))

        self.model.save(save_fname)
        timer.stop()
        return save_fname

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' %
              (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(
            save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname,
                            monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    
