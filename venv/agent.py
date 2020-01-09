class Agent(object):
    def __init__(self, total_action,
                 learning_rate=0.00025, input_dimension=(210, 160, 4),
                 batch_size=32, discount_factor=0.99,
                 memory_size=1024, dropout_prob=0.1,
                 epsilon=1, epsilon_decay=0.99,
                 load_path=None, logger=None):
        # Hyper parameters
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.discount_factor = discount_factor
        self.dropout_prob = dropout_prob
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.logger = logger
        self.load_path = load_path

        # Initialize memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []
        self.memory_count = 0
        self.training_count = 0

        # Initialize network model
        self.networks = DeepQNetwork(self.total_action,
                                     learning_rate=self.learning_rate,
                                     input_dimension=self.input_dimension,
                                     dropout_prob=self.dropout_prob,
                                     batch_size=self.batch_size,
                                     discount_factor=self.discount_factor,
                                     load_path=self.load_path,
                                     logger=self.logger)
        self.networks.target_model.set_weights(self.networks.model.get_weights())

    # Storing transition into memory
    def store_transition(self, state, action, reward, next, terminate):
        exp = ({'state': state,
                'action': action,
                'reward': reward,
                'next': next,
                'terminate': terminate})

        if len(self.memory) >= self.memory_size:
            self.memory.popleft()
        self.memory.append(exp)

        # Periodically log how many samples we've gathered so far
        if (len(self.memory) % 100 == 0) and (len(self.memory) < self.memory_size) and (self.logger is not None):
            self.logger.log("Gathered %d samples of %d" %
                            (len(self.memory), self.memory_size))

    # Policy
    def choose_action(self, state, testing=False):
        epsilon = 0.05 if testing else self.epsilon
        if random.random() <= epsilon:
            action = np.random.choice(self.total_action)
        else:
            q_val = self.networks.predict(state)
            action = np.argmax(q_val)
        return action

    def update_epsilon(self):
        if self.epsilon - self.epsilon_decay > 0.1:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0.1

    def get_state(self):
        return self.memory[randrange(0, len(self.memory))]['state']

    # Sampling
    def sample_exp_batch(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(self.memory[randrange(0, len(self.memory))])
        return np.asarray(batch)

    def max_q(self, state):
        q_values = self.networks.model.predict(state)
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        return np.random.choice(idxs)

    # Learning with Experience Replay
    def learn(self):
        self.training_count += 1
        print('Training #', self.training_count)
        batch = self.sample_exp_batch()
        self.networks.train(batch, self.networks.target_model)

    def quit(self):
        if self.load_path is None:
            if self.logger is not None:
                self.logger.log('Quitting...')
            self.DQN.save(append='_DQN')
            self.DQN_target.save(append='_DQN_target')
