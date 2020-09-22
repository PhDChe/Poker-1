''' 
pytorch implementation
'''



import gc

import torch
import os
import multiprocessing as mp


import rlcard
# from rlcard import models
from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent, nolimit_holdem_human_agent
# from rlcard.agents import RandomAgent
# from rlcard.utils import set_global_seed, tournament
from rlcard.utils import print_card


def run():
    torch.multiprocessing.freeze_support()
    env = rlcard.make('no-limit-holdem', config={'record_action': True, 'game_player_num': 2, 'env_num': 8, 'use_raw': True})
    # eval_env = rlcard.make('no-limit-holdem', config={'seed': 12, 'game_player_num': 2})
    # eval_env2 = rlcard.make('no-limit-holdem', config={'seed': 43, 'game_player_num': 2})
    #eval_env3 = rlcard.make('no-limit-holdem', config={'seed': 43, 'game_player_num': 2})
    # Set the iterations numbers and how frequently we evaluate the performance

    evaluate_every = 1024
    evaluate_num = 32
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
                    evaluate_with='best_response'))

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
                    evaluate_with='best_response'))

    # 7, 5 - all in junkies
    check_point_path = os.path.join('models/ivvan/cp/8/model-nfsp1.pth')
    checkpoint = torch.load(check_point_path)
    check_point_path = os.path.join('models/ivvan/cp/8/model-nfsp0.pth')
    checkpoint2 = torch.load(check_point_path)
    # for agent in agents:
    #     agent.load(checkpoint)
    agents[1].load(checkpoint)
    agents[0].load(checkpoint2)
    human = nolimit_holdem_human_agent.HumanAgent(env.action_num)
    env.set_agents([agents[0], agents[1]])

    while (True):
        print(">> Start a new game")

        
        
        trajectories, payoffs = env.run(is_training=False)
        if(len(trajectories[0]) == 0):
            # the bot folded immediately
            continue

        

        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1][-2]
        # print(final_state, 'waa')
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

if __name__ == '__main__':
    run()

# ray.init(log_to_driver=False)
# ray.init()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print(torch.cuda.is_available())
# torch.set_num_threads(1)
# os.environ['OMP_NUM_THREADS'] = '1'





