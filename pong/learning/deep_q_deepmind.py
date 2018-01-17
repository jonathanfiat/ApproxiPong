from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..mechanics  import constants as c
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils import common


class DeepQDeepmind:
    """Learn using the Deep-Q-Learning algorithm."""
    
    def __init__(self, args):
        """Create a DeepQDeepmind instance using args."""

        if args.self:
            self.l_pol = None
        else:
            self.l_pol = common.read_policy_args(args, "l")
        self.save_path, self.load_path = common.read_save_load_args(args)

        self.args = args
        self.sim = Pong()
        self.n_steps = 0
        self.ed = common.ExplorationDecay(args)
        
        self.db = ReplayDB(Pong.STATE_DIM, args.replay_mem_size)

        self.s = tf.Session()
        self.states = tf.placeholder(tf.float32,
            shape=(None, Pong.STATE_DIM), name="states")
            
        self.Q = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="", input_=self.states)
        self.Qh = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="qh_", input_=self.states)

        self.actions = tf.placeholder(tf.int32, (None,), "taken_actions")
        self.q_values = self.Q.take(self.actions)
        self.y = tf.placeholder(tf.float32, (None,), name="y")
        self.loss = tf.reduce_mean((self.y - self.q_values) ** 2)
        
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss,
            var_list=list(self.Q.vars()))
        self.assign_ops = self.Qh.assign(self.Q)
        self.n_iter = self.Q.load(self.load_path) + 1

    def step(self):
        """Perform a single Deep-Q-Learning step (approximately one for every
        simulator step)."""

        if self.db.n_items < self.args.replay_start_size:
            return

        if self.n_steps % self.args.target_network_update_frequency == 0:
            self.s.run(self.assign_ops)

        sample = self.db.sample(self.args.batch_size)
        v = self.Qh.predict_max(sample.s2)
        y = sample.r + (~sample.done * self.args.gamma * v)

        feed_dict = {self.states: sample.s1, self.actions: sample.a, self.y: y}
        self.s.run(self.train_op, feed_dict=feed_dict)
        
        self.n_steps += 1
        self.ed.next()

    def run_episode(self):
        """Run a single Pong episode, performing Deep-Q-Learning steps as
        apropriate."""

        self.sim.new_episode()

        state = self.sim.get_state()

        while not self.sim.done:
            l_a = self.l_pol.get_action(state)

            r_a = self.Q.predict_exploration(state[None, :], self.ed.epsilon)[0]

            self.sim.step(l_a, r_a)
            
            r = self.sim.reward("r", not self.args.partial)            
            state2 = self.sim.get_state()

            self.db.store(
                state[None,:],
                state2[None,:],
                np.array([r_a]),
                np.array([r]),
                np.array([self.sim.done])
            )

            state = state2

            self.step()

    def learn(self):
        """Perform the Deep-Q-Learning algorithm."""

        self.s.run(self.assign_ops)

        for self.n_iter in range(self.n_iter, self.args.n_iters):
            self.run_episode()

            if (self.n_iter + 1) % self.args.save_frequency == 0:
                self.Q.save(self.save_path, self.n_iter + 1)
                print("{:06d} / {:06d} : {l}|{draw}|{r}".format(
                    self.n_iter + 1, self.n_steps, **self.sim.score
                ))
                self.sim.reset()



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/DeepQ_Deepmind/", epi=None,
        ni=1000000, sf=1000, exploration=True, start=1.0, end=0.1,
        frame=1000000)
    
    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)

    parser.add_argument("--replay_mem_size", type=int, default=1000000)
    parser.add_argument("--replay_start_size", type=int, default=50000)

    parser.add_argument("--batch_size", "-bs", default=32, type=int)
    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--target_network_update_frequency", "-C",
        default=10000, type=int)

    args = parser.parse_args(argv)

    DQD = DeepQDeepmind(args)
    DQD.learn()
