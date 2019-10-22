import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random

import warnings

warnings.filterwarnings('ignore')

env = retro.make(game='SpaceInvaders-Atari2600')

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())