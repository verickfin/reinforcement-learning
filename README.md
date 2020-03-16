# Deep Q Learning Implementation in Atari Games
The created code is an implementation of Reinforcement Learning algorithm calledDeep Q Learning with Experience Replay by using Python and Keras. The algorithm is the result of Mnih et al. research that is described in [the original DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

The author implements the algorithm without any modification into the agent's behavior. This means that a single agent works to determine the best possible action to gather foods, avoid ghost, eat edible ghost, and obtain bonus's fruits only by a single training code.

## Acknowledgments
Be sure to cite the paper by Mnih et al. into your research if you use this code:
```
@article{mnih2015human,
    title={Human-level control through deep reinforcement learning},
    author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
    journal={Nature},
    volume={518},
    number={7540},
    pages={529--533},
    year={2015},
    publisher={Nature Research}
}
```
Thank you for my God, lecturer, friends, and family who always support me throughout the development of this project.

## Setup
To run the script you'll need the following dependencies:
- [Keras](http://keras.io/#installation)
- [OpenAI Gym](https://gym.openai.com/)  
- [PIL](http://www.pythonware.com/products/pil/)

which should all be available through Pip.

## Setup: Pycharm or any Local Machine Python IDE
No additional setup is needed, so simply clone the repo:
```
git clone https://github.com/verickfin/reinforcement-learning.git
```  
If the training's video want to be shown while the training is in progress, then
`# env.render()` must be uncommented.

## Setup: Google Colaboratory
The code could be accessed in Google Colaboratory through the link below:
```
https://colab.research.google.com/drive/1sAOkDVSg-4DzNt940mCkkCk2nV-lvjpw
```
As Google Colaboratory is a cloud based IDE, some dependencies must be installed, this dependencies ensures that the user could see the training result per episode in the form of videos. However, if showing the videos on the fly are not needed,
then the next two block of codes below does not need to be uncommented.

```Python
# !apt install xvfb -y

# !apt-get install python-opengl -y

# !pip install pyvirtualdisplay
# !pip install piglet
```  

Uncomment these codes to enables the function to show video output per episode.

```Python
# from pyvirtualdisplay import Display
# from IPython import display as ipythondisplay
# from IPython.display import clear_output

# display = Display(visible=0, size=(1400, 900))
# display.start()

# IMG_SIZE = None

# def show_video():
#    html = []
#    for mp4 in Path("videos").glob("*.mp4"):
#       video_b64 = base64.b64encode(mp4.read_bytes())
#       html.append('''<video alt="{}" autoplay
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                  </video>'''.format(mp4, video_b64.decode('ascii')))
#    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
```
Then, call the `show_video()` function by replacing `env.render()`.

## Usage

Listed below are customizable variables that could be configured in the code:
- `ALPHA`: A value between 0...1 that define the agent's learning rate. The default value is 0.0025.
- `EPSILON`: A value between 0...1 that handle the agent's exploration and exploitation behavior. The default epsilon value is 1.
- `EPSILON_DECAY`: A value between 0...1 that define the epsilon's decay rate. The default epsilon decay rate is 0.005.
- `GAMMA`: A discount factor which has value between 0...1 that handle the agent's behavior to prioritize short-term or long-term reward. The default gamma value is 0.95.
- `INPUT_DIMS`: A three dimensional array that define the CNN input dimension. By default, the input dimension will accept (4, 84, 84) array.
- `BATCH_SIZE`: The number of batch that Keras fit method will received. The default value is 32
- `MEMORY_SIZE`: The agent's memory capacity. The default capacity is 1000000.
- `MAX_EPISODE`: The number of the maximum episode the agent will played through the training session. By default, the maximum episode is set into 1000 episodes.
- `ENV_ID`: The environment identifier that define the agent's training environment. The default environment is'MsPacmanDeterministic-v4'.
- `IMG_SIZE`: A two dimensional array that define the preprocess game's frame image size. The default size of the preprocessed game frame is (84, 84).

These hyperparameters can be set by locating the `# Hyperparameters` comment in the code as shown below:


```python
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
```

To load the trainable weight into the Agent, the load path must be defined in the `Agent` class initialization:

```python
# Hyperparameters
.
.
# Setup
# Environment
.
env = Monitor(env, './videos/280220', force=True, video_callable=lambda episode: True)

agent = Agent(env.action_space.n, learning_rate=ALPHA, ....., load_path="checkpoint.h5")
```

A default training session can be run by typing:
```
python main.py -t
```  
Or by simply opening the source code in the chosen IDE and **run** it.

## Output

Output files are handled by mounting Google Drive into the cloud storage. These codes below have to be uncommented to handle the Google Drive mounting progress. Then, the user needs to redirect the output's file path to the place where the mounted Google Drive is located.

```Python
# from google.colab import drive
# drive.mount('/content/drive')
```

The following files will be produced as the outputs:  
1. **checkpoint.h5** is a checkpoint file that contain the agent's trainable weights;
2. **preprocess.png** is a picture of the preprocess game frame;  
3. **openaigym.video.0.18132.video00000N.mp4** is the recorded agent's gameplay per episode where N represents the game episode;
4. **openaigym.video.0.18132.video00000N.meta** is the meta file (JSON) of the recorded agent's gameplay per episode where N represents the game episode;
5. **avg plot_N.png** is a picture containing the average scores history graph where N depicted the number of episodes that the agent has finished;
6. **train plot_N.png** is a picture containing the scores history graph where N depicted the number of episodes that the agent has finished;

## Modifying The Output Path
While running the program, the output of the training which consists of checkpoint, video, preprocessed image files, and graph will be stored in two location. The video will be stored in (`videos/YYMMDD/`) while the others will be saved at the location of `main.py` file.

To change the checkpoint's save path, the code is located at the `DeepQNetwork` class as shown below:

 ```python
 class DeepQNetwork(object):
   .
   .
   # Parameters
   .
   .
   self.checkpoint = ModelCheckpoint("checkpoint_0.025_1_280220.h5",
                                      monitor='val_acc', save_weights_only=True)
 ```

The video's save path located below the `# Environment` comment:

 ```python
 .
 .
 def plot(x, y, z=None, mode=None, filename=None):
 .
 .
 # Setup
 # Environment
 env = gym.make(ENV_ID)
 env = Monitor(env, './videos/280220', force=True, video_callable=lambda episode: True)
 ```

 The preprocessed image could be saved by uncommented the following codes:

 ```python
 # im = Image.fromarray(np.uint8(observation))
 # if frame_counter < 1000:
 #     file_name = 'preprocess_' + str(frame_counter) + '.png'
 #     im.save(file_name)
 ```

 The graph files are created by the calling of the `plot()` function inside the `try...except` blocks at the very end of the codes as shown below:

 ```python
if real_eps >= 100 and real_eps % 100 == 0:
    try:
        plot(np.asarray(episodes), np.asarray(score_array), epsilons, 0,
        "train plot_" + str(episode_counter))
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)

    try:
        plot(np.asarray(episodes), np.asarray(score_array), epsilons, 1,
        "average plot_" + str(episode_counter))
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)
 ```
