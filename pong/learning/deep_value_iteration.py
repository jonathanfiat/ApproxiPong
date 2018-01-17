from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong, S
from ..mechanics.policies import RandomPolicy, Follow, NNPolicy
from ..utils.tf_machinery import NeuralNetwork
from ..utils.replay_db import ReplayDB
from ..utils.greenlet_learner import GreenletLearner
from ..utils import common


class DeepValueIteration:
    """Learn using a function-approximation version of the Value Iteration
    algorithm."""
    
    def __init__(self, args):
        """Create a DeepValueIteration instance using args."""

        self.args = args
        self.sim = Pong()
        self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, [50, 50, 50])
        self.db = ReplayDB(Pong.STATE_DIM, args.db_size)
        self.save_path, self.load_path = common.read_save_load_args(args)

        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,),
            name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)

        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.n_iter = self.nn.load(self.load_path)

        if args.rival_policy == "random":
            self.l_pol = RandomPolicy("l")
        elif args.rival_policy == "follow":
            self.l_pol = Follow("l")
        elif args.rival_policy == "self":
            self.l_pol = NNPolicy("l")
            self.l_pol.nn = self.nn

        if args.discrete_samples or args.discrete_world:
            n_bins = [args.n_cells, args.n_cells, 3, args.n_cells,
                args.n_cells, args.n_cells, args.n_cells, args.n_cells]
            self.d = common.Discretization(Pong.RANGES, n_bins)

    def add_data_points(self):
        """Use the simulator to create new data points."""

        n = self.args.states_per_iteration
        states = np.zeros((n, Pong.STATE_DIM), np.float32)
        states2 = np.zeros((n, Pong.STATE_DIM), np.float32)
        actions = np.zeros(n, np.int32)
        rewards = np.zeros(n, np.float32)
        done = np.zeros(n, np.bool)

        for i in range(n):
            self.sim.random_state()
            states[i] = self.sim.get_state()
            if self.args.discrete_world:
                states[i] = self.d.discretize(states[i])
                self.sim.set_state(states[i])

            l_a = self.l_pol.get_action(states[i])
            r_a = np.random.choice(Pong.NUM_ACTIONS)
            self.sim.step(l_a, r_a)
            states2[i] = self.sim.get_state()
            actions[i] = r_a
            rewards[i] = self.sim.reward("r", not self.args.partial)
            done[i] = self.sim.done

            if self.args.discrete_samples:
                states[i] = self.d.discretize(states[i])
                states2[i] = self.d.discretize(states2[i])

        if self.args.erase_left:
            states[:, [S.L_Y, S.L_VY]] = 0.
            states2[:, [S.L_Y, S.L_VY]] = 0.

        self.db.store(states, states2, actions, rewards, done)

    def q_iteration(self):
        """Perform a single q_iteration."""

        sample = self.db.sample()
        v = self.nn.predict_max(sample.s2)
        q_estimation = sample.r + (~sample.done * self.args.gamma * v)

        feed_dict = {
            self.nn.input: sample.s1,
            self.actions: sample.a,
            self.q_estimation: q_estimation
        }

        self.nn.train_in_batches(self.train_op, feed_dict,
            self.args.batches_per_iteration, self.args.batch_size)
        
        return self.nn.session.run(self.loss, feed_dict)

    def learn(self):
        """Create data and perform Q-iterations in a loop."""

        for self.n_iter in range(self.n_iter, self.args.n_iters):
            self.add_data_points()
            loss = self.q_iteration()

            if (self.n_iter + 1) % self.args.save_frequency == 0:
                self.nn.save(self.save_path, self.n_iter + 1)
                print("{:04d} : loss={:05.3f}".format(self.n_iter + 1, loss))



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = ArgumentParser()

    common.add_save_load_args(parser, "/tmp/Pong/DeepValueIteration/")

    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--partial", action='store_true', default=False)
    parser.add_argument("--gamma", "-g", default=0.99, type=float)
    
    parser.add_argument("--batches_per_iteration", "-bpi", default=100,
        type=int)
    parser.add_argument("--batch_size", "-bs", default=1000, type=int)
    parser.add_argument("--states_per_iteration", "-spi", default=10000,
        type=int)
    parser.add_argument("--n_iters", "-ni", default=1000, type=int)
    parser.add_argument("--db_size", type=int, default=100000)
    parser.add_argument("--save_frequency", "-sf", default=10, type=int)

    parser.add_argument("--discrete_samples", "-ds", action='store_true',
        default=False)
    parser.add_argument("--discrete_world", "-dw", action='store_true',
        default=False)
    parser.add_argument("--n_cells", "-nc", default=200, type=int)
    parser.add_argument("--erase_left", "-el", action='store_true',
        default=False)

    parser.add_argument("--rival_policy", "-rp",
        choices=["follow", "random", "self"], default="follow")

    args = parser.parse_args(argv)

    DVI = DeepValueIteration(args)
    DVI.learn()
