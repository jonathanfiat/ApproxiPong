from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class ActorCritic(GreenletLearner):
    """Learn using an Actor-Critic algorithm."""

    def __init__(self, args):
        """Create an ActorCritic instance using args."""

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
        a_values = self.q_values - self.v_values
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.p_network.output,
            labels = self.actions
        )
        
        self.p_loss = tf.reduce_mean(cross_entropy * a_values)
        self.p_optimizer = tf.train.AdamOptimizer(args.p_learning_rate)
        self.p_train_op = self.p_optimizer.minimize(self.p_loss,
            var_list=list(self.p_network.vars()))

        self.q_loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        self.q_optimizer = tf.train.AdamOptimizer(args.q_learning_rate)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss,
            var_list=list(self.q_network.vars()))

        self.s.run(tf.global_variables_initializer())
        self.nn = self.p_network
        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def critic_iteration_td0(self):
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

    def critic_iteration_td1(self):
        """Perform the "Q-Evaluation" part of the algorithm."""

        self.q_network.reinit()

        samples = self.db.iter_samples(self.args.q_sample_size,
            self.args.q_learning_iters)

        for sample in samples:
            feed_dict = {
                self.states: sample.s1,
                self.actions: sample.a,
                self.q_estimation: sample.r
            }
            self.q_network.train_in_batches(self.q_train_op, feed_dict,
                self.args.q_num_batches, self.args.q_batch_size)

        self.db.clear()

    def ciritic_iteration(self):
        if self.args.td1:
            self.critic_iteration_td1()
        else:
            self.critic_iteration_td0()

    def actor_iteration(self, states, actions, rewards):
        """Perform the "policy gradient" part of the algorithm."""

        feed_dict = {self.states: states, self.actions: actions}
        self.s.run(self.p_train_op, feed_dict)

    def iteration(self, results):
        """Perform a single iteration."""
        if not self.args.td1:
            self.db.store_episodes_results(results)
        states, actions, rewards = [], [], []
        for r in results:
            for k in range(r.rewards.shape[1] - 2, -1, -1):
                r.rewards[:, k] += self.args.gamma * r.rewards[:, k + 1]
            for i in range(r.states.shape[0]):
                states.append(r.states[i])
                actions.append(r.actions[i])
                rewards.append(r.rewards[i])
        if self.args.td1:
            self.db.store_episodes_results(results)

        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)

        self.critic_iteration()
        self.actor_iteration(states, actions, rewards)
        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions randomly using current NN parameters."""

        return self.nn.predict_random(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/ActorCritic/", ni=10000,
        exploration=True, end=0., frame=0)

    parser.add_argument("--gamma", "-g", default=0.999, type=float)
    parser.add_argument("--p_learning_rate", "-plr", default=0.001, type=float)

    parser.add_argument("--q_learning_rate", "-qlr", default=0.001, type=float)
    parser.add_argument("--q_num_batches", "-qnb", default=10, type=int)
    parser.add_argument("--q_batch_size", "-qbs", default=1000, type=int)
    parser.add_argument("--q_learning_iters", "-qli", default=100, type=int)
    parser.add_argument("--q_sample_size", "-qss", default=0, type=int)
    parser.add_argument("--no_replay", action="store_true", default=False)
    parser.add_argument("--db_size", type=int, default=10000000)

    parser.add_argument("--td1", action="store_true")

    args = parser.parse_args(argv)

    AC = ActorCritic(args)
    AC.learn()
