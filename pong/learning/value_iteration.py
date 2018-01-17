from argparse import ArgumentParser
from pathlib import Path
from time import time, clock

import numpy as np

from ..mechanics.pong import Pong, S
from ..mechanics import constants as c
from ..utils import common


TRAIN_DIMS = np.array([S.BALL_X, S.BALL_Y, S.BALL_VY, S.R_Y])
EFFECTIVE_DIM = len(TRAIN_DIMS)
DEFAULT_DIMS = np.array([S.BALL_VX, S.L_Y, S.L_VY, S.R_VY])
DEFUALT_VALUES = np.array([c.VX, 0., 0., 0.])


class ValueIteration:
    """Learn using value iteration on a discretized version of the simulator.

    Note: it is based on the false assumptions that the discretized model is
    still deterministic and markovic.
    """

    def __init__(self, args=None, ranges=None, n_bins=None, load_path=None):
        """Create a ValueIteration instance using args.
        ranges is a list of pairs (min value, max value) for each dimentions.
        n_bins is a list of ints telling how many bins to have in each
            dimention.
        """

        self.args = args
        if args is not None:
            self.save_path, self.load_path = common.read_save_load_args(args)
            load_path = self.load_path

        if load_path is None:
            assert(len(ranges) == EFFECTIVE_DIM)
            assert(len(n_bins) == EFFECTIVE_DIM)

            self.n_bins = n_bins
            self.ranges = ranges
            self.d = common.Discretization(self.ranges, self.n_bins)
            self.n = np.product(self.n_bins)

            # the size of those arrays is n+1 to account for a terminal
            # state. It allows a more efficient implementation of
            # ValueItaration.iteration.
            self.V = np.zeros(self.n + 1, np.float16)
            self.next_state = np.zeros(self.n + 1,
                (np.int32, Pong.NUM_ACTIONS))
            self.next_reward = np.zeros(self.n + 1,
                (np.float16, Pong.NUM_ACTIONS))
        else:
            with np.load(load_path) as f:
                self.n_bins = f["n_bins"]
                self.ranges = f["ranges"]
                self.d = common.Discretization(self.ranges, self.n_bins)
                self.V = f["V"]
                self.next_state = f["next_state"]
                self.next_reward = f["next_reward"]

    def fill_map(self):
        """Create a map (state, action)->(next state). It is computed once
        in order to speedup the following calculations.
        """

        sim = Pong(max_steps=None)
        s = sim.empty_state()
        s[DEFAULT_DIMS] = DEFUALT_VALUES

        # Optimization issues:
        next_state = self.next_state
        next_reward = self.next_reward
        d = self.d

        # Make the terminal state a self-loop
        next_state[self.n] = self.n

        t0 = clock()
        for i in range(0, self.n, 1000000):
            for j in range(i, min(i + 1000000, self.n)):
                s[TRAIN_DIMS] = d.index_to_state(j)
                for a in c.ACTIONS:
                    sim.fast_set_and_step(s, c.A_STAY, a)
                    if sim.hit == "r":
                        next_reward[j, a] = 1
                        next_state[j, a] = -1
                    elif sim.miss == "r":
                        next_reward[j, a] = -1
                        next_state[j, a] = -1
                    else:
                        next_state[j, a] = d.state_to_index(sim.s[TRAIN_DIMS])
            print(i, clock() -  t0)

    def iteration(self):
        """Perform a single Value-Iteration iteration."""
        
        Vnew = (self.V[self.next_state] * self.args.gamma +
                    self.next_reward).max(1)
        diff = abs(self.V - Vnew).max()
        self.V = Vnew

        return diff

    def learn(self):
        """Perform iteration in a loop."""

        for i in range(self.args.n_iters):
            diff = self.iteration()

            if diff < self.args.epsilon:
                self.save(self.save_path, i)
                break
            elif (i + 1) % self.args.save_frequency == 0:
                self.save(self.save_path, i)

    def save(self, save_path, step):
        """Save the current value estimations."""
    
        p = Path(save_path) / "{:04d}".format(step)
        p.mkdir(parents=True, exist_ok=True)
        fname = str(p / "data.npz")
        np.savez_compressed(fname,
            n_bins=self.n_bins,
            ranges=self.ranges,
            V=self.V,
            next_state=self.next_state,
            next_reward=self.next_reward
        )



#===============================================================================
# Main
#===============================================================================


def main(argv):
    parser = ArgumentParser()

    common.add_save_load_args(parser, "/tmp/Pong/ValueIteration/")

    parser.add_argument("--gamma", "-g", default=0.9, type=float)
    parser.add_argument("--epsilon", "-e", default=1e-4, type=float)
    parser.add_argument("--save_frequency", "-sf", default=100, type=int)
    parser.add_argument("--n_iters", "-ni", default=1000, type=int)

    args = parser.parse_args(argv)

    ranges = [(-1., 1.), (-1., 1.), (-0.2, 0.2), (-1., 1.)]
    n_bins = [100, 100, 100, 100]

    VI = ValueIteration(args, ranges, n_bins)
    VI.fill_map()
    print("done map")
    VI.learn()
