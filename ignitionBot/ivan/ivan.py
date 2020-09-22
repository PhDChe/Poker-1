import rlcard
from rlcard.agents.nfsp_agent_1v1 import NFSPAgent
from rlcard.agents import RandomAgent, nolimit_holdem_human_agent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger, print_card, exploitability
from multiprocessing import Pool
import os
import time


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def nfsp():
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    #os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # Make environment
    env = rlcard.make('no-limit-holdem', config={'record_action': False, 'game_player_num': 2})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': 12, 'game_player_num': 2})
    eval_env2 = rlcard.make('no-limit-holdem', config={'seed': 43, 'game_player_num': 2})

    # Set the iterations numbers and how frequently we evaluate the performance


    # The intial memory size
    memory_init_size = 1000

    # The paths for saving the logs and learning curves
    log_dir = './experiments/nolimit_holdem_nfsp_result/1v1MCNFSPv3'

    # Set a global seed
    set_global_seed(0)



    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    evaluate_every = 1000
    evaluate_num = 250
    episode_num = 5000

    # The intial memory size
    memory_init_size = 1500

    # Train the agent every X steps
    train_every = 256
    agents = []
    with graph.as_default():
        
        # Model1v1V3cp10good
        agents.append(NFSPAgent(sess,
                            scope='nfsp' + str(0),
                            action_num=env.action_num,
                            state_shape=env.state_shape,
                            hidden_layers_sizes=[512,512],
                            anticipatory_param=0.1,
                            rl_learning_rate=.1,
                            min_buffer_size_to_learn=memory_init_size,
                            q_replay_memory_init_size=memory_init_size,
                            train_every = train_every,
                            q_train_every=train_every,
                            q_mlp_layers=[512,512]))

        agents.append(NFSPAgent(sess,
                            scope='nfsp' + str(1),
                            action_num=env.action_num,
                            state_shape=env.state_shape,
                            hidden_layers_sizes=[512,512],
                            anticipatory_param=0.075,
                            rl_learning_rate=0.075,
                            min_buffer_size_to_learn=memory_init_size,
                            q_replay_memory_init_size=memory_init_size,
                            train_every = train_every//2,
                            q_train_every=train_every//2,
                            q_mlp_layers=[512,512]))

    check_point_path = os.path.join('models\\nolimit_holdem_nfsp\\1v1MCNFSPv3\\cp\\10')
    print('-------------------------------------------------------------------------------------')
    print(check_point_path)
    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
    
            global_step = tf.Variable(0, name='global_step', trainable=False)
            random_agent = RandomAgent(action_num=eval_env2.action_num)

            #easy_agent = nfsp_agents[0]
            print(agents)
            # print(nfsp_agents)
            env.set_agents(agents)
            eval_env.set_agents(agents)
            eval_env2.set_agents([agents[0], random_agent])

            # Initialize global variables
            sess.run(tf.global_variables_initializer())

            # Init a Logger to plot the learning curve
            logger = Logger(log_dir)

            for episode in range(episode_num):

                # First sample a policy for the episode
                for agent in agents:
                    agent.sample_episode_policy()
                table = []
                # Generate data from the environment
                trajectories, _ = env.run(is_training=True)

                # Feed transitions into agent memory, and train the agent
                for i in range(env.player_num):
                    for ts in trajectories[i]:
                        agents[i].feed(ts, table)

                # Evaluate the performance. Play with random agents.
                if episode % evaluate_every == 0:
                    logger.log('\n\n\n---------------------------------------------------------------\nTournament ' + str(episode/evaluate_every))
                    res = tournament(eval_env, evaluate_num)
                    res2 = tournament(eval_env2, evaluate_num//4)
                    logger.log_performance(env.timestep, res[0])
                    logger.log_performance(env.timestep, res2[0])
                    logger.log('' + str(episode_num) + " - " + str(episode) + '\n')
                    logger.log('\n\n----------------------------------------------------------------')
                    
                if episode % (evaluate_every) == 0 and not episode == 0:
                    save_dir = 'models/nolimit_holdem_nfsp/1v1MCNFSPv3/cp/10/good'+str(episode//evaluate_every)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(save_dir, 'model'))

            logger.log('\n\n\n---------------------------------------------------------------\nTournament ' + str(episode/evaluate_every))
            res = tournament(eval_env, evaluate_num)
            logger.log_performance(env.timestep, res[0])
            logger.log('' + str(episode_num) + " - " + str(episode))

            # Close files in the logger
            logger.close_files()

            # Plot the learning curve
            logger.plot('NFSP')
            
            # Save model
            save_dir = 'models/nolimit_holdem_nfsp/1v1MCNFSPv3/cp/10/good'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(save_dir, 'model'))


def play():
    import tensorflow as tf
    # We have a pretrained model here. Change the path for your model.

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    #os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # Make environment
    env = rlcard.make('no-limit-holdem', config={'record_action': True, 'game_player_num': 2})

    # Set a global seed
    set_global_seed(0)

    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 2048

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        agents = []
        agents.append(NFSPAgent(sess,
                            scope='nfsp' + str(0),
                            action_num=env.action_num,
                            state_shape=env.state_shape,
                            hidden_layers_sizes=[512,512],
                            anticipatory_param=0.1,
                            rl_learning_rate=.1,
                            min_buffer_size_to_learn=memory_init_size,
                            q_replay_memory_init_size=memory_init_size,
                            train_every = train_every,
                            q_train_every=train_every,
                            q_mlp_layers=[512,512]))

        agents.append(NFSPAgent(sess,
                            scope='nfsp' + str(1),
                            action_num=env.action_num,
                            state_shape=env.state_shape,
                            hidden_layers_sizes=[512,512],
                            anticipatory_param=0.1,
                            rl_learning_rate=.1,
                            min_buffer_size_to_learn=memory_init_size,
                            q_replay_memory_init_size=memory_init_size,
                            train_every = train_every//2,
                            q_train_every=train_every//2,
                            q_mlp_layers=[512,512]))

    check_point_path = os.path.join('models\\nolimit_holdem_nfsp\\1v1MCNFSPv2\\cp\\44')
    print('-------------------------------------------------------------------------------------')
    print(check_point_path)
    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

    
    human = nolimit_holdem_human_agent.HumanAgent(env.action_num)
    env.set_agents([human, agents[0]])

    while (True):
        print(">> Start a new game")


        trajectories, payoffs = env.run(is_training=False)
        if(len(trajectories[0]) == 0):
            # the bot folded immediately
            continue

        
    
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1][-2]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print('===============     CFR Agent    ===============')
        print_card(env.get_perfect_information()['hand_cards'][1])

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")







# # Evaluate the performance. Play with random agents.
# evaluate_num = 10000
# random_agent = RandomAgent(env.action_num)
# 
# 





nfsp()
# play()
