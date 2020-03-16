# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sAOkDVSg-4DzNt940mCkkCk2nV-lvjpw
"""
#
# !apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg- dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
#
# !apt install xvfb -y
#
# !apt-get install python-opengl -y
#
# !pip install pyvirtualdisplay
# !pip install piglet

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.initializers import VarianceScaling
from random import random, randrange
import random

import gym
import matplotlib.pyplot as plt

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
# from stable_baselines import PPO2
#
# from google.colab import drive
# drive.mount('/content/drive')
#

from keras.callbacks import *
# !ls /content/drive/'My Drive'/'Colab Notebooks'/


from PIL import Image
# from pyvirtualdisplay import Display
# from IPython import display as ipythondisplay
# from IPython.display import clear_output
from gym.wrappers import Monitor

# display = Display(visible=0, size=(1400, 900))
# display.start()
#
# IMG_SIZE = None
#
# def show_video():
#     html = []
#     for mp4 in Path("videos").glob("*.mp4"):
#         video_b64 = base64.b64encode(mp4.read_bytes())
#         html.append('''<video alt="{}" autoplay
#                       loop controls style="height: 400px;">
#                       <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                  </video>'''.format(mp4, video_b64.decode('ascii')))
#     ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def to_csv(filename, row):
    try:
        f = open(filename, 'a')
    except IOError:
        print('Logger:to_csv IO error while opening file', filename)
        return
    if type(row) is list:
        string = ','.join([str(val) for val in row])
    elif type(row) is str:
        string = row
    else:
        string = str(row)  # Try to convert it anyway
    string = string + '\n' if not string.endswith('\n') else ''
    f.write(string)
    f.close()


class DeepQNetwork(object):
    def __init__(self, total_action, learning_rate=0.00025,
                 input_dimension=(210, 160, 4),
                 batch_size=32, discount_factor=0.99,
                 load_path=None):

        # Parameters
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.input_dimension = input_dimension
        self.load_path = load_path
        self.checkpoint = ModelCheckpoint("checkpoint_0.025_1_280220.h5", monitor='val_acc', save_weights_only=True)
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.callbacks_list = [self.checkpoint]

        # Init DQN
        self.model = self.build_network()
        self.target_model = self.build_network()

        if load_path is not None:
            self.model.load_weights(self.load_path)

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
                         input_shape=self.input_dimension,))

        model.add(Flatten())
        model.add(Dense(512,
                        activation='relu',
                        kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Dense(units=self.total_action,
                        kernel_initializer=VarianceScaling(scale=2.0)))

        if self.load_path is not None:
            model.load_weights(self.load_path)

        model.compile(RMSprop(self.learning_rate),
                      loss='mean_squared_error',
                      metrics=['acc'])
        model.summary()
        return model

    def train(self, batch, target):
        s_train, q_train, train_result = [], [], []

        # Main Algorithm
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

        train_result = self.model.fit(s_train, q_train,
                       batch_size=self.batch_size, epochs=1,
                       verbose=0, callbacks=self.callbacks_list)
        # to_csv("history.csv", [train_result.history['loss'][0], train_result.history['acc'][0]])

        return train_result

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)


class Agent(object):
    def __init__(self, total_action,
                 learning_rate=0.00025, input_dimension=(210, 160, 4),
                 batch_size=32, discount_factor=0.99,
                 memory_size=1024, epsilon=1,
                 epsilon_decay=0.99, load_path=None):
        # Hyper parameters
        self.total_action = total_action
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.load_path = load_path

        # Initialize memory
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []
        self.training_count = 0

        # Initialize network model
        self.networks = DeepQNetwork(self.total_action,
                                     learning_rate=self.learning_rate,
                                     input_dimension=self.input_dimension,
                                     batch_size=self.batch_size,
                                     discount_factor=self.discount_factor,
                                     load_path=self.load_path)
        self.networks.target_model.set_weights(self.networks.model.get_weights())
        
    # Storing transition into memory
    def store_transition(self, state, action, reward, next, terminate):
        exp = ({'state': state,
                'action': action, 
                'reward': reward, 
                'next': next, 
                'terminate': terminate})
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(exp)

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
        batch = self.sample_exp_batch()
        return self.networks.train(batch, self.networks.target_model)


def preprocess(observation):
    global IMG_SIZE
    # start:stop:slice, ::step back 2 step a time
    image = observation[1:176:2, ::2]
    image = Image.fromarray(image, 'RGB').convert('L').resize(IMG_SIZE)
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])


def shift(current_stack, observation):
    return np.append(current_stack[1:], [observation], axis=0)


max_mean_score = 0

def evaluate(DQA):
    global max_mean_score
    env = gym.make(ENV_ID)
    # env = Monitor(env, './videos', force=True, video_callable=lambda episode: True)
    scores = list()
    frame_counter = 0

    while frame_counter < 10000:
        obs = preprocess(env.reset())

        # Initialize the first state with the same 4 images
        current_state = np.array([obs, obs, obs, obs])
        frame_counter += 1
        score = 0
        t = 0
        live = 3
        done = False
        episode = 0

        # Start episode
        while not done or live > 0:
            action = DQA.choose_action(np.asarray([current_state]), testing=True)
            obs, reward, done, info = env.step(action)
            live = info['ale.lives']
            obs = preprocess(obs)
            current_state = shift(current_state, obs)
            score += reward
            t += 1
            frame_counter += 1

            # End episode
            if done or live < 1 or t > 5000:
                episode += 1
                scores.append([t, score])
                break

    # clear_output()
    scores = np.asarray(scores)
    max_indices = np.argwhere(scores[:, 1] == np.max(scores[:, 1])).ravel()
    max_idx = np.random.choice(max_indices)

    # Save best model
    if max_mean_score < np.mean(scores):
        max_mean_score = np.mean(scores)
        agent.networks.model.save_weights(str(scores[len(scores)-1]) + '_best.h5')

    return scores[max_idx, :].ravel()


def plot(x, y, z=None, mode=None, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    max_score = np.max(y)
    min_score = np.min(y)
    max_index = np.where(y == max_score)
    min_index = np.where(y == min_score)

    # ax | x = eps, yl = epsilon, bl
    # ax | x = eps, yr = scores, yo
    ax.plot(x, z, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if mode == 0:
        # The Scores History Scatter Graph
        print('train:', max_score, ' ', min_score)
        print('train:', max_index, ' ', min_index)

        max_score_str = 'Max Score=' + str(max_score)
        min_score_str = 'Min Score=' + str(min_score)
        plt.text(0, max_score, max_score_str)
        plt.text(0, max_score - 300, min_score_str)
        ax2.scatter(x, y, color="C1")
    elif mode == 1:
        # The Average Scores History Scatter Graph
        N = len(y)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(y[max(0, t - 20):(t + 1)])

        max_avg_score = np.max(running_avg)
        min_avg_score = np.min(running_avg)
        print('avg:', max_avg_score, ' ', min_avg_score)

        max_avg_index = np.where(y == max_avg_score)
        min_avg_index = np.where(y == min_avg_score)
        print('avg:', max_avg_index, ' ', min_avg_index)

        max_avg_score_str = 'Max Score=' + str(max_avg_score)
        min_avg_score_str = 'Min Score=' + str(min_avg_score)
        plt.text(0, max_avg_score, max_avg_score_str)
        plt.text(0, max_avg_score - 80, min_avg_score_str)
        ax2.scatter(x, running_avg, color="C1")

    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    ax.set_title(filename)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

# Hyperparameters
ALPHA = 0.0025
EPSILON = 0.95
EPSILON_DECAY = 0.005
GAMMA = 0.95
INPUT_DIMS = (4, 84, 84)
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
MAX_EPISODE = 500
ENV_ID = 'MsPacmanDeterministic-v4'

IMG_SIZE = (84, 84)

# Setup
# Environment
env = gym.make(ENV_ID)
env = Monitor(env, './videos/280220', force=True, video_callable=lambda episode: True)
agent = Agent(env.action_space.n, learning_rate=ALPHA, 
              input_dimension=INPUT_DIMS, batch_size=BATCH_SIZE,
              discount_factor=GAMMA, memory_size=MEMORY_SIZE, epsilon=EPSILON,
              epsilon_decay=EPSILON_DECAY, load_path="checkpoint_0.025_1_270220.h5")
#"[811. 440.]_best.h5"

print('Action space:', env.action_space.n, '\n',
      'Action meaning:', env.unwrapped.get_action_meanings(), '\n',
      'Observation space:', env.observation_space.shape, '\n')

# Containers
train_result = []
score_array = []
epsilons = []

# Evaluation
t_eval = []
score_eval = []
eps_eval = []

# Counters
episode_counter = 0
frame_counter = 0

# Main
while episode_counter < MAX_EPISODE:
    score = 0
    observation = preprocess(env.reset())
    stacked_frames = np.array([observation, observation, observation, observation])
    # stacked_frames = np.reshape([stacked_frames], (1, 84, 84, 4))

    timestep = 0
    done = False
    frame_counter += 1
    live = 3
    reward = 0
    info = []
    # print('Episode:', episode_counter)

    while not done or live > 0:
        # Choose an action
        action = agent.choose_action(np.asarray([stacked_frames]))
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)

        # im = Image.fromarray(np.uint8(observation))
        # if frame_counter < 1000:
        #     file_name = 'preprocess_' + str(frame_counter) + '.png'
        #     im.save(file_name)

        next_state = shift(stacked_frames, observation)
        live = info['ale.lives']

        frame_counter += 1

        # Store experience
        clip = np.clip(reward, -1, 1)
        agent.store_transition(np.asarray([stacked_frames]), action,
                               clip, np.asarray([next_state]), done)

        # Train
        if len(agent.memory) >= 1000:
            if timestep % 4 == 0:
                agent.learn()

        if agent.training_count % 10000 == 0 and agent.training_count >= 10000:
            agent.networks.target_model.set_weights(agent.networks.model.get_weights())
            agent.update_epsilon()

        # Eval scores
        if frame_counter % 10000 == 0:
            t, s = evaluate(agent)
            t_eval.append(t)
            score_eval.append(s)
            eps_eval.append(episode_counter)

        stacked_frames = next_state
        score += reward
        timestep += 1

    score_array.append(score)
    epsilons.append(agent.epsilon)

    print('Score:', score)
    # print('Lives:', info['ale.lives'], 'Done:', ydone)
    print('Eps:', episode_counter, 'Training #', agent.training_count)

    real_eps = episode_counter + 1
    episodes = [i + 1 for i in range(real_eps)]

    if real_eps >= 100 and real_eps % 100 == 0:
        try:
            plot(np.asarray(episodes), np.asarray(score_array), epsilons, 0, "train plot_" + str(episode_counter))
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)

        try:
            plot(np.asarray(episodes), np.asarray(score_array), epsilons, 1, "average plot_" + str(episode_counter))
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)

    episode_counter += 1

print('Max_mean_score:', max_mean_score)
