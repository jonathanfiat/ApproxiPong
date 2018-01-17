from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong, random_sign, S
from ..mechanics  import constants as c
from ..utils.tf_machinery import NeuralNetwork
from ..utils import common


class ImitationLearner:
    """Learn by imitating an expert."""

    def __init__(self, args):
        """Create an ImitationLearner using args."""
        
        self.args = args
        self.save_path, self.load_path = common.read_save_load_args(args)
        
        if not self.args.decomposed:
            self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS,
                [50, 50, 50])
            self.y = tf.placeholder(tf.int32, (None,),
                name="taken_actions")
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.nn.output,
                    labels=self.y
            ))
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self.nn.output, self.y, 1),
                tf.float32
            ))

        else:
            self.nn = NeuralNetwork(Pong.STATE_DIM, 1, [50, 50, 50])
            self.y = tf.placeholder(tf.float32, (None,),
                name="target")
            self.loss = tf.reduce_mean((self.nn.output[:, 0] - self.y)**2)
            diffs_true = self.nn.input[:, S.R_Y] - self.y
            diffs_pred = self.nn.input[:, S.R_Y] - self.nn.output[:,0]
            
            up = []
            down = []
            stay = []
            for diffs in [diffs_true, diffs_pred]:
                up.append(diffs < (0.5 * c.HPL))
                down.append(diffs > (0.5 * c.HPL))
                stay.append(~(up[-1] | down[-1]))
            agree = (
                (up[0] & up[1]) |
                (down[0] & down[1]) |
                (stay[0] & stay[1])
            )
            self.accuracy = tf.reduce_mean(tf.cast(agree, tf.float32))

        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.nn.session.run(tf.global_variables_initializer())

        self.n_iter = self.nn.load(self.args.load_path)

    def set_data(self, states, labels):
        """Split generated data into "train" and "test" datasets."""
    
        p = np.random.permutation(self.args.train_size + self.args.test_size)
        test = p[:self.args.test_size]
        train = p[self.args.test_size:]
        self.test = {
            self.nn.input: states[test],
            self.y: labels[test],
        }
        self.train = {
            self.nn.input: states[train],
            self.y: labels[train],
        }

    def create_data(self):
        """Simulate data."""
    
        sim = Pong()
        l_pol = common.read_policy_args(self.args, "l")
        r_pol = common.read_policy_args(self.args, "r")
        m = self.args.train_size + self.args.test_size

        states = np.zeros((m, sim.STATE_DIM), np.float32)
        actions = np.zeros(m, np.int32)
        targets = np.zeros(m, np.float32)
            
        for i in range(m):
            if self.args.artificial:
                sim.random_state()
            elif sim.done:
                sim.new_episode()
            
            state = sim.get_state()
            l_a = l_pol.get_action(state)
            r_a = r_pol.get_action(state)
            
            states[i] = state
            actions[i] = r_a
            targets[i] = r_pol.get_target(state)

            sim.step(l_a, r_a)

        if not self.args.decomposed:
            self.set_data(states, actions)
        else:
            self.set_data(states, targets)

    def learn(self):
        """Learn by imitation."""
    
        for i in range(self.n_iter, self.args.n_iters):
            self.nn.train_in_batches(self.train_op, self.train,
                self.args.batches_per_iteration, self.args.batch_size)
            p = self.nn.save(self.args.save_path, i + 1)
            train_eval = self.nn.accuracy(self.accuracy, self.train,
                self.args.batch_size)
            test_eval = self.nn.accuracy(self.accuracy, self.test,
                self.args.batch_size)
            print("Iteration {}: train={:05.3f}, test={:05.3f}".format(i + 1,
                train_eval, test_eval))



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = ArgumentParser()

    common.add_policy_args(parser, "l", ["follow", "predict"], "follow")
    common.add_policy_args(parser, "r", ["follow", "predict"], "predict")
    common.add_save_load_args(parser, "/tmp/Pong/Imitation/")

    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)

    parser.add_argument("--train_size", type=int, default=200000)
    parser.add_argument("--test_size", type=int, default=100000)

    parser.add_argument("--batch_size", "-bs", default=600, type=int)
    parser.add_argument("--batches_per_iteration", "-bpi", default=500,
        type=int)
    parser.add_argument("--n_iters", "-ni", default=100, type=int)

    parser.add_argument("--artificial", action="store_true", default=False)
    parser.add_argument("--decomposed", action="store_true", default=False)

    args = parser.parse_args(argv)

    IL = ImitationLearner(args)
    print("Generating data...")
    IL.create_data()
    print("Learning from data...")
    IL.learn()
