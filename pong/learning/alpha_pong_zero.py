from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from ..mechanics.pong import Pong
from ..mechanics  import constants as c
from ..utils.tf_machinery import NeuralNetwork
from ..utils.greenlet_learner import GreenletLearner, EpisodeResult
from ..utils import common
from ..utils import one_sided_mcts
from ..utils import two_sided_mcts


class AlphaPongZero(GreenletLearner):
    """Learn using an extremely simplified version of the Alpha-Go-Zero
    algorithm, dubbed "Alpha-Pong-Zero".
    """

    def __init__(self, args):
        """Create an AlphaPongZero instance using args."""

        super().__init__(args)

        self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, [50, 50, 50])
        self.search_probabilities = tf.placeholder(tf.float32,
            (None, self.nn.output_dim), name="mcts_probs")
        self.evaluation = tf.tanh(self.nn.affine(
            "evaluation",
            self.nn.layers[-1],
            1,
            relu=False
        )[:, 0])
        self.reward = tf.placeholder(tf.float32, (None,), name="reward")
        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.nn.output, labels=self.search_probabilities
        ))
        self.evaluation_loss = tf.reduce_mean(
            (self.reward - self.evaluation)**2
        )
        self.total_loss = self.evaluation_loss + self.cross_entropy_loss
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

        self.n_iter = self.nn.load(self.load_path)
        self.ed = common.ExplorationDecay(args, self.n_iter)

    def prior_one(self, state):
        """A prior function for the single player case."""

        a = self.main.switch(state)
        return a[:-1], a[-1]

    def prior_two(self, state):
        """A prior function for the self-play case."""

        a_l = self.main.switch(Pong.flip_state(state))
        a_r = self.main.switch(state)
        return np.vstack((a_l[:-1], a_r[:-1])), a_r[-1]

    def run_episode_opponent(self, *args):
        """Simulate a single episode against a predefined policy."""

        sim = Pong(random_positions=True)
        m = one_sided_mcts.MCTS(sim, self.l_pol, self.prior_one,
            self.args.mcts_depth, self.args.mcts_c_puct)

        states = np.zeros((1, self.args.episode_max_length, Pong.STATE_DIM),
            np.float32)
        probs = np.zeros((1, self.args.episode_max_length, Pong.NUM_ACTIONS),
            np.float32)

        for t in range(self.args.episode_max_length):
            m.search(self.args.mcts_simulations)
            p = m.root.probabilities()

            states[0, t] = m.root.state
            probs[0, t] = p
            t += 1

            a = np.random.choice(Pong.NUM_ACTIONS, 1, p=p)[0]
            m.step(a)

            if m.done():
                break

        rewards = np.zeros((1, t), np.float32)
        if m.done():
            rewards[:] = m.root

        w = {0: "draw", 1:"r", -1:"l"}[int(rewards[0, 0])]
        return EpisodeResult(states[:, :t], probs[:, :t], rewards[:, :t], w)

    def run_episode_self(self, *args):
        """Simulate a single episode in a self-play scenario."""

        sim = Pong(random_positions=True)
        m = two_sided_mcts.MCTS(sim, self.prior_two,
            self.args.mcts_depth, self.args.mcts_c_puct)

        states = np.zeros((2, self.args.episode_max_length, Pong.STATE_DIM),
            np.float32)
        probs = np.zeros((2, self.args.episode_max_length, Pong.NUM_ACTIONS),
            np.float32)

        for t in range(self.args.episode_max_length):
            m.search(self.args.mcts_simulations)

            r_state = m.root.state
            l_state = Pong.flip_state(r_state)
            p_l, p_r = m.root.probabilities()

            states[0, t] = r_state
            probs[0, t] = p_r
            states[1, t] = l_state
            probs[1, t] = p_l
            t += 1

            l_a = np.random.choice(Pong.NUM_ACTIONS, 1, p=p_l)[0]
            r_a = np.random.choice(Pong.NUM_ACTIONS, 1, p=p_r)[0]
            m.step(l_a, r_a)

            if m.done():
                break

        rewards = np.zeros((2, t), np.float32)
        if m.done():
            rewards[0, :] = -m.root
            rewards[1, :] = m.root

        w = {0: "draw", 1:"r", -1:"l"}[int(rewards[0, 0])]
        return EpisodeResult(states[:, :t], probs[:, :t], rewards[:, :t], w)

    def iteration(self, results):
        """Perform a single Alpha-Pong-Zero iteration."""

        states, probs, rewards = [], [], []
        for result in results:
            for i in range(result.states.shape[0]):
                states.append(result.states[i])
                probs.append(result.actions[i])
                rewards.append(result.rewards[i])
        states = np.concatenate(states)
        probs = np.concatenate(probs)
        rewards = np.concatenate(rewards)

        feed_dict = {
            self.nn.input: states,
            self.search_probabilities: probs,
            self.reward: rewards
        }

        self.nn.train_in_batches(self.train_op, feed_dict,
            self.args.batches_per_iteration, self.args.batch_size)
        
        self.ed.next()
        return {"states": states, "probs": probs, "rewards": rewards}
    
    def decide_actions(self, eval_states, *args):
        """Return the prior probabilities and evaluations of eval_states."""

        feed_dict = {self.nn.input: eval_states}
        P, V = self.nn.session.run([self.nn.probabilities, self.evaluation],
            feed_dict)

        e = self.ed.epsilon
        P = P * (1 - e) + (1 / Pong.NUM_ACTIONS) * e
        V = V * (1 - e)

        return np.hstack([P, V[:, None]])


#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = common.standard_parser("/tmp/Pong/AlphaPongZero/", epi=100, sf=1)
    common.add_exploration_args(parser, start=1., end=0., frame=0)
    
    parser.add_argument("--learning_rate", "-lr", default=lr, type=float)
    parser.add_argument("--batches_per_iteration", "-bpi", default=bpi,
        type=int)
    parser.add_argument("--batch_size", "-bs", default=bs, type=int)

    parser.add_argument("--mcts_simulations", type=int, default=100)
    parser.add_argument("--mcts_depth", type=int, default=50)
    parser.add_argument("--mcts_c_puct", type=float, default=1.4)
    parser.add_argument("--episode_max_length", type=int, default=c.MAX_STEPS)

    args = parser.parse_args(argv)

    APZ = AlphaPongZero(args)
    APZ.learn()
