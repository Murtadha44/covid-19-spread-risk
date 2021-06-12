from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np

def model_fun(network):
    if network['num_fc_layers']==2:
        model_m = Sequential()
        if network['num_lstm_layers'] == 1:
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, input_shape=(network["x_input_size"]),
                     activity_regularizer=l2(network['reg_r'])))
        elif network['num_lstm_layers'] == 2:
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, input_shape=(network["x_input_size"]),
                     activity_regularizer=l2(network['reg_r'])))
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, dropout=network['dropout_prob'],
                     activity_regularizer=l2(network['reg_r'])))
        elif network['num_lstm_layers'] == 3:
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, input_shape=(network["x_input_size"]),
                     activity_regularizer=l2(network['reg_r'])))
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, dropout=network['dropout_prob'],
                     activity_regularizer=l2(network['reg_r'])))
            model_m.add(
                LSTM(units=network['num_lstm_states'], return_sequences=True, dropout=network['dropout_prob'],
                     activity_regularizer=l2(network['reg_r'])))
        model_m.add(Dense(network['num_fc_nodes'], activation='linear',
                          activity_regularizer=l2(network['reg_r'])))
        model_m.add(ReLU())
        model_m.add(Dropout(network['dropout_prob']))
        model_m.add(Dense(network['num_out_nodes'], kernel_initializer='normal'))
        print(model_m.summary())

        model_m.compile(loss='mean_squared_error',
                    optimizer=Adam(lr = network['base_learning_rate'],decay=network['decayRate']), metrics=['mae'])
    else:
        raise NotImplementedError("Architecture does not exist")
    return model_m


class DataGenerator(tf.keras.utils.Sequence): # original https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size, shuffle):
        'Initialization'
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indeces):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = self.data[indeces]
        y = self.labels[indeces]

        # adding random start point, its max is X.shape[1]-num_days_req
        num_days_req = 14
        rand_start = np.random.randint(0, X.shape[1]-num_days_req)
        rand_end = np.random.randint(rand_start+num_days_req, X.shape[1])
        X = X[:, rand_start:rand_end, :]
        y = y[:, rand_start:rand_end, :]
        return X, y
