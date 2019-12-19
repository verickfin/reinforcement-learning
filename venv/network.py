import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from keras.initializers import VarianceScaling
from collections import deque


class DeepQNetwork(object):
    def __init__(self, total_action, learning_rate,
                 fcl_dims=1024, input_dimension=(210, 160, 4)):
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.fcl_dims = fcl_dims
        self.input_dimension = input_dimension

        # Init DQN
        self.model = self.build_network()
        self.model.compile(RMSprop(learning_rate), 'mean_squared_error')
        self.q_target_model = self.build_network()
        self.q_target_model.compile(RMSprop(learning_rate), 'mean_squared_error')
        self.q_target_model.set_weights(self.model.get_weights())

    def build_network(self):
        # CNN
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, activation='relu', input_shape=self.input_dimension,
                         kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Conv2D(64, kernel_size=4, activation='relu',
                         kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Conv2D(128, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.fcl_dims, activation='relu',
                        kernel_initializer=VarianceScaling(scale=2.0)))

        model.add(Dense(units=self.total_action, kernel_initializer=VarianceScaling(scale=2.0)))

        return model

    def fit(self, state, q_values):
        self.model.fit(input, state, q_values, epochs=1, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def predict_target(self, state):
        return self.q_target_model.predict(state)

    def save(self, path):
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)


class Agent(object):
    def __init__(self, learning_rate, discount_factor, epsilon, total_action, batch_size,
                 memory_size=20000, input_dims=(210, 160, 4)):
        # Hyper parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.99
        self.total_action = total_action
        self.input_dims = input_dims

        # Initialize memory
        self.batch_size = batch_size
        self.memory = deque()
        self.memory_count = 0
        self.memory_size = memory_size

        # Initialize network model
        self.network = DeepQNetwork(self.total_action, self.learning_rate)

    # Storing memory
    def store_transition(self, state, action, reward, done, next_state):
        exp = (state, action, reward, done, next_state)
        if self.memory_count < self.memory_size:
            self.memory_count += 1
        else:
            self.memory.popleft()
        self.memory.append(exp)

    # Policy
    def choose_action(self, state):
        random = np.random.random()
        if random <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.run(self.q_eval.q_values, feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        return action

    def sample_exp_batch(self, batch_size):
        max_memory = self.memory_counter if self.memory_counter < self.memory_size else self.memory_size

        batch = np.random.choice(self.memory, max_memory)

        state_batch = np.array([i[0] for i in batch])
        action_batch = np.array([i[1] for i in batch])
        reward_batch = np.array([i[2] for i in batch])
        terminal_state_batch = np.array([i[3] for i in batch])
        next_state_batch = np.array([i[4] for i in batch])

        return state_batch, action_batch, reward_batch, terminal_state_batch, next_state_batch

    # Learning with Experience Replay
    def learn(self):
        state, action, reward, done, next_state = self.sample_exp_batch(self.memory_size)

        q_current = self.network.predict(state)
        q_next = self.network.predict(next_state)
        q_target = self.network.predict_target(state)

        for i in range(state.shape[0]):
            if done[i]:
                q_current[i, action[i]] = reward[i]
            else:
                next_best_action = np.argmax(q_next[i, :])
                q_current[i, action[i]] = reward[i] + self.discount_factor * q_target[i, next_best_action]

        self.network.fit(state, q_current)

        if self.memory_counter > 10000:
            if self.epsilon > 0.05:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0.05

    def save(self):
        self.q_eval.save()
        self.q_next.save()

    def load(self):
        self.q_eval.load()
        self.q_next.load()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.run(tf.assign(t, e))
