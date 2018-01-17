from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..utils.tf_machinery import NeuralNetwork
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class SuccessLearning(GreenletLearner):
    """Learn using the SuccessLearning algoritm."""

    def __init__(self, args):
        """Create a PolicyGradient instance using args."""

        super().__init__(args)

        self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, [50, 50, 50])
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.rewards = tf.placeholder(tf.float32, (None,), "rewards")
        self.ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.nn.output,
            labels = self.actions
        )
        self.loss = tf.reduce_mean(self.ce_loss * self.rewards)
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def iteration(self, results):
        """Perform a single policy gradient iteration on episodes' results."""

        states, actions, rewards = [], [], []
        for r in results:
            for k in range(r.rewards.shape[1] - 2, -1, -1):
                r.rewards[:, k] += self.args.gamma * r.rewards[:, k + 1]
            for i in range(r.states.shape[0]):
                states.append(r.states[i])
                actions.append(r.actions[i])
                rewards.append(r.rewards[i])
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)

        flags = rewards > 0.

        feed_dict = {
            self.nn.input: states[flags],
            self.actions: actions[flags],
            self.rewards: rewards[flags],
        }

        self.nn.train_in_batches(self.train_op, feed_dict,
            self.args.batches_per_iteration, self.args.batch_size)
        self.ed.next()

    def decide_actions(self, eval_states, *args):
        """Select actions randomly using current NN parameters."""

        return self.nn.predict_random(eval_states, self.ed.epsilon)



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/SuccessLearning/",
        exploration=True, start=1., end=0., frame=0)

    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--batches_per_iteration", "-bpi", default=1000,
        type=int)
    parser.add_argument("--batch_size", "-bs", default=1000, type=int)
    parser.add_argument("--gamma", "-g", default=0.999, type=float)

    args = parser.parse_args(argv)
    SL = SuccessLearning(args)    
    SL.learn()
