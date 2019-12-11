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
        self.q_next = DeepQNetwork('q_next', total_action, learning_rate,
                                   input_dimension=input_dims, checkpoint_dir=q_next_dir)
        self.q_eval = DeepQNetwork('q_eval', total_action, learning_rate,
                                   input_dimension=input_dims, checkpoint_dir=q_eval_dir)

        self.action_space = [i for i in range (self.total_actions)]
        self.action_memory = np.zeros((self.memory_size, self.total_actions), dtype=int8)
        self.reward_memory = np.zeros(self.memory_size)
        self.state_memory = np.zeros(self.memory_size, *input_dims)
        self.next_state_memory = np.zeros(self.memory_size, self.total_actions)
        self.terminal_state_memory = np.zeros(self.memory_size, dtype=int8)

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
            actions = self.q_eval.sess.run(self.q_eval.q_values, feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory_counter % self.replace_target == 0:
            self.update_graph()

        max_memory = self.memory_counter if self.memory_counter < self.memory_size else self.memory_size

        batch = np.random.choice(max_memory, self.batch_size)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        terminal_state_batch = self.terminal_state_memory[batch]
        next_state_batch = self.next_state_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.q_values, feed_dict={self.q_eval.input: state_batch})
        q_next = self.q_next.sess.run(self.q_next.q_values, feed_dict={self.q_next.input: next_state_batch})

        q_target = q_eval.copy()
        index = np.arange(self.batch_size)
        q_target[index, action_indices] = reward_batch + self.discount_factor *\
                                          np.max(q_next, axis=1) * terminal_state_batch

        network = self.q_eval.sess.run(self.q_eval.train,
                                       feed_dict={self.q_eval.input: state_batch,
                                                  self.q_eval.action_space: next_state_batch,
                                                  self.q_eval.q_target: q_target})

        if self.memory_counter > 10000:
            if self.epsilon > 0.05:
                self.epsilon -= 4e-7
            elif self.epsilon <= 0.05:
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
            self.q_eval.sess.run(tf.assign(t, e))
