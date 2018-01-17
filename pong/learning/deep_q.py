from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class DeepQ(GreenletLearner):
    """Learn using a simplified version of the Deep-Q-Learning algorithm."""

    def __init__(self, args):
        """Create a DeepQ instance using args."""

        super().__init__(args)

        self.db = ReplayDB(Pong.STATE_DIM, args.db_size)
        self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, [50, 50, 50])

        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,),
            name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def iteration(self, results):
        """Perform a single Deep-Q-Learning iteration."""

        self.db.store_episodes_results(results)

        samples = self.db.iter_samples(self.args.q_sample_size,
            self.args.q_learning_iters)
        for sample in samples:
            v = self.nn.predict_max(sample.s2, self.args.batch_size)
            q = sample.r + (~sample.done * self.args.gamma * v)

            feed_dict = {
                self.nn.input: sample.s1,
                self.actions: sample.a,
                self.q_estimation: q
            }

            self.nn.train_in_batches(self.train_op, feed_dict,
                self.args.num_batches, self.args.batch_size)

        if self.args.no_replay:
            self.db.clear()
        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions using an episilon-greedy policy based on current NN
        parameters."""

        return self.nn.predict_exploration(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/DeepQ/", exploration=True)

    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--batch_size", "-bs", default=1000, type=int)
    parser.add_argument("--num_batches", "-qnb", default=1000, type=int)
    parser.add_argument("--q_learning_iters", "-qli", default=1, type=int)
    parser.add_argument("--q_sample_size", "-qss", default=0, type=int)
    parser.add_argument("--no_replay", action='store_true', default=False)
    parser.add_argument("--db_size", type=int, default=10000000)

    args = parser.parse_args(argv)

    DQ = DeepQ(args)
    DQ.learn()
    
