import ray
import numpy as np
#import tensorflow as tf


@ray.remote
class Actor(object):
    def __init__(self, env):
        import tensorflow as tf
        import numpy as np
        from rlcard.agents.mc_nfspV1 import NFSPAgent
        from rlcard.agents import RandomAgent, nolimit_holdem_human_agent
        from rlcard.utils import set_global_seed, tournament
        from rlcard.utils import Logger, print_card, exploitability
        self.env = env
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
