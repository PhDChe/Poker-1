from rlcard.utils import *
import gc
# from rlcard.agents import mc_nfspV1

class Env(object):
    '''
    The base Env class. For all the environments in RLCard,
    we should base on this class and implement as many functions
    as we can.
    '''

    def __init__(self, config):
        ''' Initialize the environment

        Args:
            config (dict): A config dictionary. All the fields are
                optional. Currently, the dictionary includes:
                'seed' (int) - A environment local random seed.
                'env_num' (int) - If env_num>1, the environemnt wil be run
                  with multiple processes. Note the implementatino is
                  in `vec_env.py`.
                'allow_step_back' (boolean) - True if allowing
                 step_back.
                'allow_raw_data' (boolean) - True if allow
                 raw obs in state['raw_obs'] and raw legal actions in
                 state['raw_legal_actions'].
                'single_agent_mode' (boolean) - True if single agent mode,
                 i.e., the other players are pretrained models.
                'active_player' (int) - If 'singe_agent_mode' is True,
                 'active_player' specifies the player that does not use
                  pretrained models.
                There can be some game specific configurations, e.g., the
                number of players in the game. These fields should start with
                'game_', e.g., 'game_player_num' we specify the number of
                players in the game. Since these configurations may be game-specific,
                The default settings shpuld be put in the Env class. For example,
                the default game configurations for Blackjack should be in
                'rlcard/envs/blackjack.py'
                TODO: Support more game configurations in the future.
        '''
        self.allow_step_back = self.game.allow_step_back = config['allow_step_back']
        self.allow_raw_data = config['allow_raw_data']
        self.record_action = config['record_action']
        if self.record_action:
            self.action_recorder = []

        # Game specific configurations
        # Currently only support blackjack
        # TODO support game configurations for all the games
        supported_envs = ['blackjack']
        if self.name in supported_envs:
            _game_config = self.default_game_config.copy()
            for key in config:
                if key in _game_config:
                    _game_config[key] = config[key]
            self.game.configure(_game_config)

        # Get the number of players/actions in this game
        self.player_num = self.game.get_player_num()
        self.action_num = self.game.get_action_num()

        # A counter for the timesteps
        self.timestep = 0

        # Modes
        self.single_agent_mode = config['single_agent_mode']
        self.active_player = config['active_player']

        # Load pre-trained models if single_agent_mode=True
        if self.single_agent_mode:
            self.model = self._load_model()
            # If at least one pre-trained agent needs raw data, we set self.allow_raw_data = True
            for agent in self.model.agents:
                if agent.use_raw:
                    self.allow_raw_data = True
                    break

        # Set random seed, default is None
        self._seed(config['seed'])


    def reset(self):
        '''
        Reset environment in single-agent mode
        Call `_init_game` if not in single agent mode
        '''
        if not self.single_agent_mode:
            return self._init_game()

        while True:
            state, player_id = self.game.init_game()
            while not player_id == self.active_player:
                self.timestep += 1
                action, _ = self.model.agents[player_id].eval_step(self._extract_state(state))
                if not self.model.agents[player_id].use_raw:
                    action = self._decode_action(action)
                state, player_id = self.game.step(action)

            if not self.game.is_over():
                break

        return self._extract_state(state)

    def step(self, action, raw_action=False):
        ''' Step forward

        Args:
            action (int): The action taken by the current player
            raw_action (boolean): True if the action is a raw action

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        '''
        if not raw_action:
            action = self._decode_action(action)
        if self.single_agent_mode:
            return self._single_agent_step(action)
        # print('-----')
        # print(action)
        # print(self.get_player_id())
        # print('-----')
        self.timestep += 1
        # Record the action for human interface
        if self.record_action:
            self.action_recorder.append([self.get_player_id(), action])
        
        next_state, player_id = self.game.step(action)
        # print(next_state.pre)
        return self._extract_state(next_state), player_id

    def step_back(self):
        ''' Take one step backward.

        Returns:
            (tuple): Tuple containing:

                (dict): The previous state
                (int): The ID of the previous player

        Note: Error will be raised if step back from the root node.
        '''
        if not self.allow_step_back:
            raise Exception('Step back is off. To use step_back, please set allow_step_back=True in rlcard.make')

        if not self.game.step_back():
            return False
        
        player_id = self.get_player_id()
        state = self.get_state(player_id)

        return state, player_id

    def set_agents(self, agents):
        '''
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        '''
        if self.single_agent_mode:
            raise ValueError('Setting agent in single agent mode or human mode is not allowed.')

        self.agents = agents
        # print(agents)
        # If at least one agent needs raw data, we set self.allow_raw_data = True
        for agent in self.agents:
            if agent.use_raw:
                self.allow_raw_data = True
                break

    def run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        if self.single_agent_mode:
            raise ValueError('Run in single agent not allowed.')

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        
        # print('hi')
        while not self.is_over():
            

        
            # if(len(table)>=5):
            #     print(table)
            if not is_training:
                # print('hello')
                action, _ = self.agents[player_id].eval_step(state) 
            else:
                action = self.agents[player_id].step(state)
                # print('hola')
                


            # Environment steps
            # print('--------------------')
            # print(action)
            # print('--------------------')
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)
    
            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)
        return trajectories, payoffs

    def is_over(self):
        ''' Check whether the curent game is over

        Returns:
            (boolean): True if current game is over
        '''
        return self.game.is_over()

    def get_player_id(self):
        ''' Get the current player id

        Returns:
            (int): The id of the current player
        '''
        return self.game.get_player_id()


    def loadState(self, state):
        """
        load the game state from the state dict
        """
        print(state)

    def get_state(self, player_id):
        ''' Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (numpy.array): The observed state of the player
        '''
        return self._extract_state(self.game.get_state(player_id))

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            (list): A list of payoffs for each player.

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.np_random = self.np_random
        return seed

    def _init_game(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        state, player_id = self.game.init_game()
        if self.record_action:
            self.action_recorder = []
        return self._extract_state(state), player_id

    def _load_model(self):
        ''' Load pretrained/rule model

        Returns:
            model (Model): A Model object
        '''
        raise NotImplementedError

    def _extract_state(self, state):
        ''' Extract useful information from state for RL. Must be implemented in the child class.

        Args:
            state (dict): The raw state

        Returns:
            (numpy.array): The extracted state
        '''
        raise NotImplementedError


    def _decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def _get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list): A list of legal actions' id.

        Note: Must be implemented in the child class.
        '''
        raise NotImplementedError

    def _single_agent_step(self, action):
        ''' Step forward for human/single agent

        Args:
            action (int): The action takem by the current player

        Returns:
            next_state (numpy.array): The next state
        '''
        reward = 0.
        done = False
        self.timestep += 1
        state, player_id = self.game.step(action)
        while not self.game.is_over() and not player_id == self.active_player:
            self.timestep += 1
            action, _ = self.model.agents[player_id].eval_step(self._extract_state(state))
            if not self.model.agents[player_id].use_raw:
                action = self._decode_action(action)
            state, player_id = self.game.step(action)

        if self.game.is_over():
            reward = self.get_payoffs()[self.active_player]
            done = True
            state = self.reset()
            return state, reward, done

        return self._extract_state(state), reward, done

    @staticmethod
    def init_game():
        ''' (This function has been replaced by `reset()`)
        '''
        raise ValueError('init_game is removed. Please use env.reset()')
