import gym
from copy import deepcopy

# import numpy as np
import random
from rlcard.games.nolimitholdem import Game

from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import NFSPAgentPytorch as NFSPAgent

from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import remove_illegal
from rlcard.envs.nolimitholdem import NolimitholdemEnv
import rlcard

from Env.AtariEnv.AtariEnvWrapper import make_atari_env
from Env.PokerEnv.PokerState import PokerState

# To allow easily extending to other tasks, we built a wrapper on top of the 'real' environment.
class EnvWrapper():
    def __init__(self, env_name, max_episode_length = 0, enable_record = False, record_path = "1.mp4"):
        self.env_name = env_name

        self.env_type = None
        print('wtf')

        self.env = rlcard.make('no-limit-holdem', config={'record_action': True, 'game_player_num': 2, 'seed': 477})
        # self.state, self.pointer = self.game.init_game() 
        
        memory_init_size = 300

        # The paths for saving the logs and learning curves
        self.log_dir = './experiments/nolimit_holdem_nfsp_result/ivvan'

        # Set a global seed
        


        self.evaluate_every = 512
        self.evaluate_num = 64
        self.episode_num = 20480

        # The intial memory size
        self.memory_init_size = 256

        # Train the agent every X steps
        self.train_every = 256
        self.agents = []

        self.agents.append(NFSPAgent(
                        scope='nfsp' + str(0),
                        action_num=self.env.action_num,
                        state_shape=self.env.state_shape,
                        hidden_layers_sizes=[512,512],
                        anticipatory_param=0.1,
                        rl_learning_rate=0.015,
                        sl_learning_rate=0.0075,
                        q_epsilon_start=.3,
                        min_buffer_size_to_learn=memory_init_size,
                        q_replay_memory_size=20480,
                        q_replay_memory_init_size=memory_init_size,
                        train_every = self.train_every+44,
                        q_train_every=self.train_every,
                        q_mlp_layers=[512,512],
                        evaluate_with='average_policy'))

        self.agents.append(NFSPAgent(
                        scope='nfsp' + str(1),
                        action_num=self.env.action_num,
                        state_shape=self.env.state_shape,
                        hidden_layers_sizes=[512,512],
                        anticipatory_param=0.1,
                        rl_learning_rate=0.015,
                        sl_learning_rate=0.0075,
                        q_epsilon_start=.3,
                        q_replay_memory_size=20480,
                        min_buffer_size_to_learn=memory_init_size,
                        q_replay_memory_init_size=memory_init_size,
                        train_every = self.train_every+44,
                        q_train_every=self.train_every,
                        q_mlp_layers=[512,512],
                        evaluate_with='average_policy'))

        


        self.env.set_agents(self.agents)
        self.env.reset()
        #initialize env to be equal to the game
        # print(self.state)
        # self.env = PokerState(self.state['hand'], self.state['public_cards'], 250 - self.state['all_chips'][0], 250 - self.state['all_chips'][1], abs(self.state['all_chips'][0] - self.state['all_chips'][1]), self.state['all_chips'][0] + self.state['all_chips'][1], self.state['all_chips'][0], self.state['all_chips'][1])
        self.action_n = 6

        self.max_episode_length = self.env._max_episode_steps if max_episode_length == 0 else max_episode_length

        self.current_step_count = 0

        self.since_last_reset = 0

    def reset(self):
        state = self.env.reset()
        self.current_step_count = 0
        self.since_last_reset = 0

        return self.env.game

    def step(self, action):
        # print(action)
        
        # next_state, playerid = self.env.step(action)
        self.env.step(action)

        return self.env.game, self.env.clone_state().run()

    def eval(self):
        return self.env.clone_state().run()

    def restore(self, state):
        # keep same
        # print(state, '----------------------------this is the state-------')
        self.env.load_state(state)
        
        
        return self.env.game

    def getReward(self):
        return self.env.get_result(self.env)

    def checkpoint(self):
        return self.env.clone_state()

    def get_action_n(self):
        return self.action_n

    def get_max_episode_length(self):
        return self.max_episode_length


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
