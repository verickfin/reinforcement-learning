import os
import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from keras.initializers import VarianceScaling


class DeepQNetwork(object):
    def __init__(self, total_action, learning_rate=0.00025,
                 input_dimension=(210, 160, 4), dropout_prob=0.1,
                 batch_size=32, discount_factor=0.99,
                 load_path=None, logger=None):
        # Parameters
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.training_file = 'training.csv'
        self.logger = logger
        self.input_dimension = input_dimension
        self.load_path = load_path

        # Init DQN
        self.model = self.build_network()
        self.target_model = self.build_network()

        if load_path is not None:
            self.model.load_weights(self.load_path)

        if self.logger is not None:
            self.logger.to_csv(self.training_file, 'Loss,Accuracy')

    # Clear
    def build_network(self):
        # CNN
        model = Sequential()
        model.add(Conv2D(32,
                         kernel_size=8,
                         strides=(4, 4),
                         activation='relu',
                         data_format='channels_first',
                         input_shape=self.input_dimension,
                         kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Conv2D(64,
                         kernel_size=4,
                         strides=(2, 2),
                         activation='relu',
                         data_format='channels_first',
                         input_shape=self.input_dimension,
                         kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Conv2D(64,
                         kernel_size=3,
                         strides=(1, 1),
                         activation='relu',
                         data_format='channels_first',
                         input_shape=self.input_dimension, ))

        model.add(Flatten())
        model.add(Dense(512,
                        activation='relu',
                        kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Dense(units=self.total_action,
                        kernel_initializer=VarianceScaling(scale=2.0)))

        if self.load_path is not None:
            self.logger.log('Loading weights from file...')
            model.load_weights(self.load_path)

        model.compile(RMSprop(self.learning_rate),
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def train(self, batch, target):
        s_train, q_train = [], []

        for experience in batch:
            s_train.append(experience['state'].astype(np.float64))

            next_state = experience['next'].astype(np.float64)
            next_state_predict = target.predict(next_state).ravel()
            next_q = np.max(next_state_predict)

            q_list = list(self.predict(experience['state'])[0])
            if not experience['terminate']:
                q_list[experience['action']] = experience['reward'] + self.discount_factor * next_q
            else:
                q_list[experience['action']] = experience['reward']
            q_train.append(q_list)

        s_train = np.asarray(s_train).squeeze()
        q_train = np.asarray(q_train).squeeze()

        r = self.model.fit(s_train, q_train, batch_size=self.batch_size, epochs=1, verbose=2)

        if self.logger is not None:
            self.logger.to_csv(self.training_file, [r.history['loss'][0], r.history['acc'][0]])

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=''):
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
        self.model.save_weights(self.logger.path + f)