ALPHA = 0.00025
EPSILON = 1
EPSILON_DECAY = 9e-7
GAMMA = 0.99
INPUT_DIMS = (4, 110, 84)
BATCH_SIZE = 32
MEMORY_SIZE = 1024
MAX_EPISODE = 50000
DROPOUT_PROB = 0.
ENV_ID = 'MsPacmanDeterministic-v4'

IMG_SIZE = (84, 110)

if __name__ == '__main__':
    # Setup
    logger = Logger(debug=True, append=ENV_ID)

    # Environment
    env = gym.make(ENV_ID)
    env = Monitor(env, './videos', force=True, video_callable=lambda episode: True)
    agent = Agent(env.action_space.n, learning_rate=ALPHA,
                  input_dimension=INPUT_DIMS, dropout_prob=DROPOUT_PROB,
                  batch_size=BATCH_SIZE, discount_factor=GAMMA,
                  memory_size=MEMORY_SIZE, epsilon=EPSILON,
                  epsilon_decay=EPSILON_DECAY, load_path=None, logger=logger)
    logger.log({
        'Action space': env.action_space.n,
        'Observation space': env.observation_space.shape
    })

    # Containers
    score_array = []
    mean_array = []
    state_array = []
    q_array = []

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
        print('Episode:', episode_counter)
        while not done:
            print('Timestep:', timestep)
            show_video()
            # Choose an action
            action = agent.choose_action(np.asarray([stacked_frames]))
            print('Action:', action)
            print('Stacked Frames[0]: ', stacked_frames[0])

            # Get next observation area and shift the stacked frames
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            print('Obs length:', len(observation))
            next_state = shift(stacked_frames, observation)

            frame_counter += 1

            # Store experience
            clip = np.clip(reward, -1, 1)
            agent.store_transition(np.asarray([stacked_frames]), action,
                                   clip, np.asarray([next_state]), done)

            # Train
            if len(agent.memory) >= 500:
                if timestep % 4 == 0:
                    agent.learn()

                if agent.training_count % 500 == 0 and agent.training_count >= 500:
                    agent.networks.target_model.set_weights(agent.networks.model.get_weights())

                if agent.training_count % 500 == 0 and agent.training_count >= 500:
                    logger.to_csv(test_csv,
                                  [np.mean(score_array), np.mean(mean_array)])
                    del score_array[:]
                    del mean_array[:]

                agent.update_epsilon()

            stacked_frames = next_state
            score += reward

            timestep += 1

            if frame_counter % 2000 == 0:
                t_evaluation, score_evaluation = evaluate(agent)

            if len(state_array) < 30:
                for x in range(random.randint(1, 5)):
                    state_array.append(agent.memory[randrange(0, len(agent.memory))]['state'])
            else:
                score_array.append(score)
                q_array = [agent.max_q(state) for state in state_array]
                mean_array.append(np.mean(q_array))

        episode_counter += 1

    logger.log(evaluate(agent))