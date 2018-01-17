from argparse import ArgumentParser
from itertools import product

import numpy as np


def add_save_load_args(parser, default):
    parser.add_argument("--save_path", "-sp", default=default)
    parser.add_argument("--load_path", "-lp", default=None)
    parser.add_argument("--cont", "-c", action='store_true', default=False)


def read_save_load_args(args):
    if args.cont:
        if args.load_path is not None:
            load_path = args.load_path 
        else:
            load_path = args.save_path
    else:
        load_path = None
    return args.save_path, load_path


def add_policy_args(parser, side, choices, default="follow"):
    if side == "l":
        parser.add_argument("--left", "-l", choices=choices, default=default)
        parser.add_argument("--left_args", "-la", default="")
    else:
        parser.add_argument("--right", "-r", choices=choices, default=default)
        parser.add_argument("--right_args", "-ra", default="")


def read_policy_args(args, side):
    from ..mechanics import policies

    if side == "l":
        p = args.left
        a = args.left_args
    else:
        p = args.right
        a = args.right_args

    return policies.POLICIES[p].create_from_commandline(side, a)


def standard_parser(path, epi=1000, bs=1000, ni=1000, sf=100,
                exploration=False, start=1.0, end=0.01, frame=6):

    parser = ArgumentParser()
    
    add_policy_args(parser, "l", ["follow", "predict"])
    add_save_load_args(parser, path)

    if epi is not None:
        parser.add_argument("--episodes_per_iteration", "-epi", default=epi,
            type=int)
    parser.add_argument("--n_iters", "-ni", default=ni, type=int)
    parser.add_argument("--save_frequency", "-sf", default=sf, type=int)
    parser.add_argument("--partial", action='store_true', default=False)
    parser.add_argument("--self", action='store_true', default=False)
    
    if exploration:
        parser.add_argument("--epsilon_start", type=float, default=start)
        parser.add_argument("--epsilon_end", type=float, default=end)
        parser.add_argument("--epsilon_frame", type=int, default=frame)

    return parser


def add_exploration_args(parser, start=1.0, end=0.01, frame=20):
    parser.add_argument("--epsilon_start", type=float, default=start)
    parser.add_argument("--epsilon_end", type=float, default=end)
    parser.add_argument("--epsilon_frame", type=int, default=frame)


class ExplorationDecay:
    """A simple linearly decaying parameter."""

    def __init__(self, args, step_num=0):
        """Create an ExplorationDecay instance using args."""

        self.start = args.epsilon_start
        self.end = args.epsilon_end
        self.frame = args.epsilon_frame
        self.step_num = step_num
        self.next()
    
    def next(self):
        self.epsilon = self(self.step_num)
        self.step_num += 1
    
    def __call__(self, n):
        """Get the exploration parameter for step n."""

        if n >= self.frame:
            return self.end
        else:
            a = self.start - self.end
            r = n / self.frame
            return self.start - r * a


class Discretization:
    def __init__(self, ranges, n_bins):
        assert(len(ranges) == len(n_bins))
        self.dim = len(ranges)
        self.n_bins = np.array(n_bins, np.int64)
        self.ranges = ranges
        self.bins = []
        self.translations = []
        for r, n in zip(self.ranges, self.n_bins):
            borders = np.linspace(r[0], r[1], n - 1)
            mid = np.zeros(n)
            mid[[0, -1]] = r
            mid[1:-1] = (borders[:-1] + borders[1:]) / 2.
            self.bins.append(borders)
            self.translations.append(mid)
        self.prod = np.zeros(self.dim, np.int64)
        for i in range(self.dim):
            self.prod[i] = np.product(self.n_bins[i+1:])
        
        self.sphere = None
    
    def set_sphere(self, dist=1):
        self.sphere = np.array(list(product(*[list(range(-dist, dist + 1))] * self.dim)))
        
    def state_to_index(self, state):
        """Translate a continuous state into a discrete representation."""

        return sum(p * np.searchsorted(b, s)
            for p, s, b in zip(self.prod, state, self.bins))

    def index_to_state(self, index):
        """Translate the discrete representation of a state into the average
        state it represets."""

        return np.array([t[(index // p) % n]
            for p, n, t in zip(self.prod, self.n_bins, self.translations)])

    def discretize(self, obs):
        """Get the discretization of obs."""

        return np.array([t[np.searchsorted(b, s)]
            for t, s, b in zip(self.translations, obs, self.bins)])

    def state_to_neighborhood(self, state):
        """Translate a continuous state into a discrete representation."""

        base = np.array([np.searchsorted(b, s) for
            p, s, b in zip(self.prod, state, self.bins)])        
        multi_ind = base + self.sphere
        legal = (multi_ind >= 0).all(1) & (multi_ind < self.n_bins).all(1)
        return (multi_ind[legal] * self.prod).sum(1)
