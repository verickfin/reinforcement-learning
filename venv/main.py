import os
import gym
from network import Agent, DeepQNetwork
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt

ALPHA = 0.05
EPSILON = 0.5
GAMMA = 0.99
INPUT_DIMS = (180, 160, 4)
TOTAL_ACTIONS = 3
MEMORY_SIZE = 50000
BATCH_SIZE = 64


def preprocess(observation):
    observation /= 255
    return np.mean(observationp[30:, :], axis=2).reshape(180, 60, 1)


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for i, j in enumerate(stacked_frames):
            stacked_frames[i, :] = frame
    else:
        stacked_frames[0:buffer_size - 1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size - 1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)
    return stacked_frames


def plot(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = Agent(learning_rate=ALPHA, discount_factor=GAMMA,
                  total_action=TOTAL_ACTIONS, epsilon=EPSILON,
                  batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE, input_dims=INPUT_DIMS)
    load = False
    if load:
        agent.load()
    score_array = []
    epsilon_array = []
    num_games = 1000
    stack_size = 4
    score = 0

    while agent.memory_counter < MEMORY_SIZE:
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = np.random.choice([0, 1, 2])
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
            action -= 1
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_

        print('Random game finish...')

        for i in range(num_games):
            done = False
            if i % 10 == 0 & i > 0:
                avg_scores = np.mean(scores[max(0, i - 10):(i + 1)])
                print('Episode', i, 'Score', score, 'Average Score %.3f' % avg_scores,
                      'Epsilon %.3f' % agent.epsilon)
                agent.save()
            else:
                print('Episode', i, 'Score', score)

            observation = env.reset()
            observation = preprocess(observation)
            stacked_frames = None
            observation = stack_frames(stacked_frames, observation, stack_size)
            while not done:
                action = agent.choose_action(observation)
                action += 1
                observation_, reward, done, info = env.step(action)
                observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
                action -= 1
                agent.store_transition(observation, action, reward, observation_, int(done))
                observation = observation_
                agent.learn()
                score += reward
            score_array.append(score)
            epsilon_array.append(agent.epsilon)
            x = [i + 1 for i in range(numGames)]
            plot(x, scores, eps_history, filename)
