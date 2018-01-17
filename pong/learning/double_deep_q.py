from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class DoubleDeepQ(GreenletLearner):
    """Learn using a simplified version of the Double-Deep-Q-Learning algorithm.
    """

    def __init__(self, args):
        """Create a DoubleDeepQ instance using args."""

        super().__init__(args)

        self.db = ReplayDB(Pong.STATE_DIM, args.db_size)
        self.s = tf.Session()

        self.states = tf.placeholder(tf.float32, shape=(None, Pong.STATE_DIM),
            name="states")

        self.q1_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="", input_=self.states)

        self.q2_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="q2_", input_=self.states)

        self.q_estimation = tf.placeholder(tf.float32, (None,),
            name="q_estimation")
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q1_values = self.q1_network.take(self.actions)
        self.q2_values = self.q2_network.take(self.actions)

        self.q1_loss = tf.reduce_mean(
            (self.q_estimation - self.q1_values) ** 2
        )
        self.q2_loss = tf.reduce_mean(
            (self.q_estimation - self.q2_values) ** 2
        )

        self.q1_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.q2_optimizer = tf.train.AdamOptimizer(args.learning_rate)

        self.q1_train_op = self.q1_optimizer.minimize(self.q1_loss,
            var_list=list(self.q1_network.vars()))
        self.q2_train_op = self.q2_optimizer.minimize(self.q2_loss,
            var_list=list(self.q2_network.vars()))

        self.nn = self.q1_network
        self.n_iter = self.q1_network.load(self.load_path)

        self.ed = common.ExplorationDecay(args, self.n_iter)

    def iteration(self, results):
        """Perform a single Double-Deep-Q-Learning iteration."""

        self.db.store_episodes_results(results)

        samples = self.db.iter_samples(self.args.q_sample_size,
            self.args.q_learning_iters)
        for sample in samples:
            a1 = self.q1_network.predict_argmax(sample.s2, self.args.batch_size)
            v1 = self.s.run(self.q2_values, {self.states: sample.s2,
                self.actions: a1})
            q1 = sample.r + (~sample.done * self.args.gamma * v1)

            a2 = self.q2_network.predict_argmax(sample.s2, self.args.batch_size)
            v2 = self.s.run(self.q1_values, {self.states: sample.s2,
                self.actions: a2})
            q2 = sample.r + (~sample.done * self.args.gamma * v2)
            
            feed_dict = {self.states: sample.s1, self.actions: sample.a}

            feed_dict[self.q_estimation] = q1
            self.q2_network.train_in_batches(self.q2_train_op, feed_dict,
                self.args.num_batches, self.args.batch_size)

            feed_dict[self.q_estimation] = q2
            self.q1_network.train_in_batches(self.q1_train_op, feed_dict,
                self.args.num_batches, self.args.batch_size)

        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions based using an episilon-greedy policy based on current
        NN parameters."""

        return self.nn.predict_exploration(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/DoubleDeepQ/", exploration=True)

    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--batch_size", "-bs", default=1000, type=int)
    parser.add_argument("--num_batches", "-qnb", default=1000, type=int)
    parser.add_argument("--q_learning_iters", "-qli", default=1, type=int)
    parser.add_argument("--q_sample_size", "-qss", default=0, type=int)
    parser.add_argument("--no_replay", action='store_true', default=False)
    parser.add_argument("--db_size", type=int, default=100000000)

    args = parser.parse_args(argv)

    DDQ = DoubleDeepQ(args)
    DDQ.learn()
