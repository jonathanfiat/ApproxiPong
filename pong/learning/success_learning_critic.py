from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class SuccessLearningCritic(GreenletLearner):
    """Learn using the SuccessLearning algoritm."""

    def __init__(self, args):
        """Create a PolicyGradient instance using args."""

        super().__init__(args)

        super().__init__(args)
        self.db = ReplayDB(Pong.STATE_DIM, args.db_size)
        self.s = tf.Session()
        
        self.states = tf.placeholder(tf.float32,
            shape=(None, Pong.STATE_DIM), name="states")
        self.p_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="", input_=self.states)
        self.q_network = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
            [50, 50, 50], session=self.s, name_prefix="q_", input_=self.states)
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_estimation = tf.placeholder(tf.float32, (None,),
            name="q_estimation")
        
        self.q_values = self.q_network.take(self.actions)
        pq = self.p_network.probabilities * self.q_network.output
        self.v_values = tf.reduce_sum(pq, axis=1)
        self.a_values = self.q_values - self.v_values
        
        self.q_loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        self.q_optimizer = tf.train.AdamOptimizer(args.q_learning_rate)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss,
            var_list=list(self.q_network.vars()))

        self.rewards = tf.placeholder(tf.float32, (None,), "rewards")
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.p_network.output,
            labels = self.actions
        )
        self.p_loss = tf.reduce_mean(cross_entropy * self.rewards)
        self.p_optimizer = tf.train.AdamOptimizer(args.p_learning_rate)
        self.p_train_op = self.p_optimizer.minimize(self.p_loss,
            var_list=list(self.p_network.vars()))
        
        self.nn = self.p_network
        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def critic_iteration(self):
        """Perform the "Q-Evaluation" part of the algorithm."""

        self.q_network.reinit()

        samples = self.db.iter_samples(self.args.q_sample_size,
            self.args.q_learning_iters)

        for sample in samples:
            v = NeuralNetwork.run_op_in_batches(self.s, self.v_values,
                {self.states: sample.s2}, self.args.q_batch_size)
            q = sample.r + (~sample.done * self.args.gamma * v)
            feed_dict = {
                self.states: sample.s1,
                self.actions: sample.a,
                self.q_estimation: q
            }
            self.q_network.train_in_batches(self.q_train_op, feed_dict,
                self.args.q_num_batches, self.args.q_batch_size)

        if self.args.no_replay:
            self.db.clear()

    def actor_iteration(self, states, actions):
        """Perform the "policy gradient" part of the algorithm."""

        feed_dict = {self.states: states, self.actions: actions}

        a = NeuralNetwork.run_op_in_batches(self.s, self.a_values, feed_dict,
            self.args.p_batch_size)

        flags = a > 0.

        feed_dict = {
            self.nn.input: states[flags],
            self.actions: actions[flags],
            self.rewards: a[flags],
        }

        self.p_network.train_in_batches(self.p_train_op, feed_dict,
            self.args.p_num_batches, self.args.p_batch_size)

    def iteration(self, results):
        """Perform a single policy gradient iteration on episodes' results."""

        self.db.store_episodes_results(results)
        states, actions = [], []
        for r in results:
            for i in range(r.states.shape[0]):
                states.append(r.states[i])
                actions.append(r.actions[i])
        states = np.concatenate(states)
        actions = np.concatenate(actions)

        self.critic_iteration()
        self.actor_iteration(states, actions)
        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions randomly using current NN parameters."""

        return self.nn.predict_random(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/SuccessLearningCritic/",
        exploration=True, start=1., end=0., frame=0)

    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--db_size", type=int, default=10000000)
    parser.add_argument("--no_replay", action='store_true', default=False)

    parser.add_argument("--q_learning_rate", "-qlr", default=0.001, type=float)
    parser.add_argument("--q_num_batches", "-qnb", default=10, type=int)
    parser.add_argument("--q_batch_size", "-qbs", default=1000, type=int)
    parser.add_argument("--q_learning_iters", "-qli", default=1000, type=int)
    parser.add_argument("--q_sample_size", "-qss", default=10000, type=int)

    parser.add_argument("--p_learning_rate", "-plr", default=0.001, type=float)
    parser.add_argument("--p_num_batches", "-pnb", default=1000, type=int)
    parser.add_argument("--p_batch_size", "-pbs", default=1000, type=int)

    args = parser.parse_args(argv)
    SLC = SuccessLearningCritic(args)    
    SLC.learn()
