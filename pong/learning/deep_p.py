from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class DeepP(GreenletLearner):
    """Learn using the Deep-P-Learning algorithm."""

    def __init__(self, args):
        """Create an DeepP instance using args."""

        super().__init__(args)
        self.db = ReplayDB(Pong.STATE_DIM, args.db_size)
        self.s = tf.Session()

        self.states = tf.placeholder(tf.float32,
            shape=(None, Pong.STATE_DIM), name="states")
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_estimation = tf.placeholder(tf.float32, (None,),
            name="q_estimation")
        self.e = tf.placeholder_with_default(0.0, ())

        self.p_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="", input_=self.states)
        self.q_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="q_", input_=self.states)

        max_values = self.q_network.take(self.p_network.output_argmax)
        mean_values = tf.reduce_mean(self.q_network.output, axis=1)
        self.v_values = (1 - self.e) * max_values + self.e * mean_values
        self.q_values = self.q_network.take(self.actions)
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)

        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss,
            var_list=list(self.q_network.vars()))
        self.assign_ops = self.p_network.assign(self.q_network)

        self.s.run(tf.global_variables_initializer())
        self.s.run(self.assign_ops)
        
        self.nn = self.p_network
        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def iteration(self, results):
        """Perform a single iteration."""

        self.db.store_episodes_results(results)
        
        self.q_network.reinit()
        
        samples = self.db.iter_samples(self.args.sample_size,
            self.args.learning_iters)
        for sample in samples:
            if self.args.td1:
                feed_dict = {
                    self.states: sample.s1,
                    self.actions: sample.a,
                    self.q_estimation: sample.r
                }
            else:
                v = NeuralNetwork.run_op_in_batches(self.s, self.v_values,
                    {self.states: sample.s2}, self.args.batch_size,
                    {self.e: self.ed.epsilon})

                q = sample.r + (~sample.done * self.args.gamma * v)
                feed_dict = {
                    self.states: sample.s1,
                    self.actions: sample.a,
                    self.q_estimation: q
                }
            self.q_network.train_in_batches(self.train_op, feed_dict,
                self.args.num_batches, self.args.batch_size)
        
        if self.args.td1:
            self.db.clear()

        self.s.run(self.assign_ops)
        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions randomly using current NN parameters."""

        return self.nn.predict_exploration(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/DeepP/", exploration=True,
        start=1., end=0.01, frame=1)

    parser.add_argument("--db_size", type=int, default=10000000)
    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--learning_rate", "-qlr", default=0.001, type=float)
    parser.add_argument("--num_batches", "-qnb", default=10, type=int)
    parser.add_argument("--batch_size", "-qbs", default=1000, type=int)
    parser.add_argument("--learning_iters", "-qli", default=1000, type=int)
    parser.add_argument("--sample_size", "-qss", default=10000, type=int)

    parser.add_argument("--td1", action="store_true")

    args = parser.parse_args(argv)

    DP = DeepP(args)
    DP.learn()
