import os
import numpy as np
import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, name, total_action, learning_rate,
                 fcl_dims=1024, input_dimension=(210, 160, 4), checkpoint_dir='temp/dqn'):
        self.name = name
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.fcl_dims = fcl_dims
        self.input_dimension = input_dimension
        self.checkpoint_dir = checkpoint_dir

        # Network Names
        self.input = None
        self.action_space = None
        self.q_target = None
        self.q_values = None
        self.loss = None
        self.train = None

        self.sess = tf.session()
        self.sess.run(tf.global_variable_initializer())
        self.build_network()

        # File Management
        self.saver = tf.train.saver()
        self.checkpoint_file = os.path.join(checkpoint_dir, 'DeepQNetwork.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        None,
                                        *self.input_dimension,
                                        name='input')
            self.action_space = tf.placeholder(tf.float32,
                                               None,
                                               self.total_action,
                                               name='action_space')
            self.q_target = tf.placeholder(tf.float32,
                                           None,
                                           self.total_action,
                                           name='q_value')

            # CNN
            conv1 = tf.layers.conv2D(inputs=self.input, filters=32, kernel_size=(8, 8),
                                     strides=4, name='conv1',
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2D(inputs=conv1_activated, filters=64, kernel_size=(4, 4),
                                     strides=2, name='conv2',
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv2_activated = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2D(inputs=conv2_activated, filters=128, kernel_size=(3, 3),
                                     strides=2, name='conv3',
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)

            flatten = tf.layers.flatten(conv3_activated)
            dense = tf.layers.dense(flat, units=self.fcl_dims, activation=tf.nn.relu,
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.q_values = tf.layers.dense(dense1, units=self.n_actions,
                                            kernel_initializer=tf.variance_scaling_initializer(scale=2))

            # Q actual values
            # self.q = tf.reduce_sum(tf.multiply(self.q_values, self.action_space))
            self.loss = tf.reduce_mean(tf.square(self.q_values, self.q_target))

            self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        def save(self):
            print("Saving Session........")
            self.saver.save(self.sess, self.checkpoint_file)

        def load(self):
            print("Restoring Session........")
            self.saver.restore(self.sess, self.checkpoint_file)

class Agent(object):
    def __init__(self, learning_rate, discount_factor, total_action, epsilon, batch_size,
                 memory_size, replace_target=1000, input_dims=(210,160,4),
                 q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.total_actions = total_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0
        self.replace_target = replace_target
        self.input_dims = input_dims
        self.q_next = DeepQNetwork(learning_rate=learning_rate, total_action=total_action, input_dimension=input_dims, )q_next_dir
        self.q_eval = q_eval_dir

        self.action_space = [i for i in range (self.total_actions)]
        self.state_memory = np.zeros(self.memory_size, *input_dims)
        self.next_state_memory = np.zeros(self.memory_size, self.total_actions)
        self.action_memory = np.zeros((self.memory_size, self.total_actions), dtype=int8)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=int8)

    def store_transition(self, state, action, reward, next_state, terminal_state):
        index = self.memory_counter % self.memory_size
        action = np.zeros(self.total_actions)
        actions[action] = 1.0
        self.state_memory = state
        self.next_state_memory = next_state
        self.action_memory = action
        self.reward_memory = reward
        self.terminal_memory = terminal_state

    def choose_action(self, state):
        random = np.random.random()
        if random < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval