import os
import numpy as np
import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, name, total_action, learning_rate,
                 input_dimension=(210,160,4), checkpoint_dir='temp/dqn'):
        self.name = name
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.checkpoint_dir = checkpoint_dir

        # Session
        self.sess = tf.session()
        self.sess.run(tf.global_variable_initializer())
        self.build_network()

        # File Management
        self.saver = tf.train.saver()
        self.checkpoint_file = os.path.join(checkpoint_dir,'DeepQNetwork.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

