''' 
pytorch implementation
'''

import gc

import torch
import os
import ray

import rlcard
from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger



ray.init(log_to_driver=False)
# ray.init()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print(torch.cuda.is_available())
# torch.set_num_threads(24)
# os.environ['OMP_NUM_THREADS'] = '24'


env = rlcard.make('no-limit-holdem', config={'record_action': True, 'game_player_num': 2, 'seed': 477})
eval_env = rlcard.make('no-limit-holdem', config={'seed': 12, 'game_player_num': 2})
eval_env2 = rlcard.make('no-limit-holdem', config={'seed': 43, 'game_player_num': 2})
#eval_env3 = rlcard.make('no-limit-holdem', config={'seed': 43, 'game_player_num': 2})
# Set the iterations numbers and how frequently we evaluate the performance


# The intial memory size
memory_init_size = 300

# The paths for saving the logs and learning curves
log_dir = './experiments/nolimit_holdem_nfsp_result/ivvan'

# Set a global seed
set_global_seed(577)


evaluate_every = 512
evaluate_num = 64
episode_num = 20480

# The intial memory size
memory_init_size = 256

# Train the agent every X steps
train_every = 256
agents = []

agents.append(NFSPAgent(
                scope='nfsp' + str(0),
                action_num=env.action_num,
                state_shape=env.state_shape,
                hidden_layers_sizes=[512,512],
                anticipatory_param=0.1,
                rl_learning_rate=0.015,
                sl_learning_rate=0.0075,
                q_epsilon_start=.3,
                min_buffer_size_to_learn=memory_init_size,
                q_replay_memory_size=20480,
                q_replay_memory_init_size=memory_init_size,
                train_every = train_every+44,
                q_train_every=train_every,
                q_mlp_layers=[512,512],
                evaluate_with='average_policy'))

agents.append(NFSPAgent(
                scope='nfsp' + str(1),
                action_num=env.action_num,
                state_shape=env.state_shape,
                hidden_layers_sizes=[512,512],
                anticipatory_param=0.1,
                rl_learning_rate=0.015,
                sl_learning_rate=0.0075,
                q_epsilon_start=.3,
                q_replay_memory_size=20480,
                min_buffer_size_to_learn=memory_init_size,
                q_replay_memory_init_size=memory_init_size,
                train_every = train_every+44,
                q_train_every=train_every,
                q_mlp_layers=[512,512],
                evaluate_with='average_policy'))

random_agent = RandomAgent(action_num=eval_env2.action_num)



env.set_agents(agents)
eval_env.set_agents([agents[0], random_agent])
eval_env2.set_agents([random_agent, agents[1]])
# eval_env3.set_agents([agents[1], random_agent])

# Initialize global variables

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):
    print(episode, end = '\r')
    #print('oh')

    # First sample a policy for the episode
    for agent in agents:
        agent.sample_episode_policy()

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for i in range(env.player_num):
        # update ray rl model
        for ts in trajectories[i]:
            agents[i].feed(ts)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log('\n\n\n---------------------------------------------------------------\nTournament ' + str(episode/evaluate_every))
        # tournament(eval_env2, 6)
        # exploitability.exploitability(eval_env, agents[0], 500)

        res = tournament(env, evaluate_num)
        logger.log_performance(env.timestep, res[0])
        res2 = tournament(eval_env, evaluate_num//3)
        logger.log_performance(env.timestep, res2[0])
        res3 = tournament(eval_env2, evaluate_num//3)
        logger.log_performance(env.timestep, res3[0])
        logger.log('' + str(episode_num) + " - " + str(episode) + '\n')
        logger.log('\n\n----------------------------------------------------------------')
        
    if episode % (evaluate_every) == 0:
        save_dir = 'models/ivvan/cp/'+str(episode//evaluate_every)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for agnt in agents:    
            state_dict = agnt.get_state_dict()
            print(state_dict.keys())
            torch.save(state_dict, os.path.join(save_dir, 'model-' +agnt._scope + '.pth'))
    

logger.log('\n\n\n---------------------------------------------------------------\nTournament ' + str(episode/evaluate_every))
res = tournament(eval_env, evaluate_num)
logger.log_performance(env.timestep, res[0])
logger.log('' + str(episode_num) + " - " + str(episode))

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('NFSP')


save_dir = 'models/ivvan/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = agent.get_state_dict()
print(state_dict.keys())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

























# # Make environment
# env = rlcard.make('leduc-holdem', config={'seed': 0})
# eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

# # Set the iterations numbers and how frequently we evaluate the performance
# evaluate_every = 100
# evaluate_num = 1000
# episode_num = 100000

# # The intial memory size
# memory_init_size = 1000

# # Train the agent every X steps
# train_every = 1

# # The paths for saving the logs and learning curves
# log_dir = './experiments/limit_holdem_dqn_result/'

# # Set a global seed
# set_global_seed(0)

# agent = DQNAgent(scope='dqn',
#                  action_num=env.action_num,
#                  replay_memory_init_size=memory_init_size,
#                  train_every=train_every,
#                  state_shape=env.state_shape,
#                  mlp_layers=[128, 128],
#                  device=torch.device('cpu'))
# random_agent = RandomAgent(action_num=eval_env.action_num)
# env.set_agents([agent, random_agent])
# eval_env.set_agents([agent, random_agent])

# # Init a Logger to plot the learning curve
# logger = Logger(log_dir)

# for episode in range(episode_num):

#     # Generate data from the environment
#     trajectories, _ = env.run(is_training=True)

#     # Feed transitions into agent memory, and train the agent
#     for ts in trajectories[0]:
#         agent.feed(ts)

#     # Evaluate the performance. Play with random agents.
#     if episode % evaluate_every == 0:
#         logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
    
#     gc.collect()

# # Close files in the logger
# logger.close_files()

# # Plot the learning curve
# logger.plot('DQN')

# # Save model
# save_dir = 'models/leduc_holdem_dqn_pytorch'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# state_dict = agent.get_state_dict()
# print(state_dict. keys())
# torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

