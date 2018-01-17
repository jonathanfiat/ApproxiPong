import argparse
import shlex

import numpy as np
import tensorflow as tf

from .pong import Pong, S
from ..learning.value_iteration import ValueIteration, TRAIN_DIMS
from ..utils.tf_machinery import NeuralNetwork
from ..utils import one_sided_mcts
from ..utils import two_sided_mcts
from ..utils import common
from . import constants as c


class Policy:
    """An ABC for policies"""

    @classmethod
    def create_from_commandline(cls, side, argv):
        raise NotImplementedError

    def __init__(self, side, name, save_path=None):
        self.side = side
        self.name = name

    def get_action(self, state, *args):
        raise NotImplementedError
    
    def new_episode(self):
        pass


class Manual(Policy):
    """A policy for human players: reads input from keyboard"""

    @classmethod
    def create_from_commandline(cls, side, argv):
        return cls(side)

    def __init__(self, side, name="Player", save_path=None):
        super().__init__(side, name)

    def get_action(self, state, buttons):
        if self.side == "l":
            if "a" in buttons:
                return c.A_UP
            if "z" in buttons:
                return c.A_DOWN
        else:
            if "up" in buttons:
                return c.A_UP
            if "down" in buttons:
                return c.A_DOWN
        return c.A_STAY


class RandomPolicy(Policy):
    """Act randomly."""

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", "-n", default="Random")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))
    
    def __init__(self, side, name="Random"):
        super().__init__(side, name)
    
    def get_action(self, state, *args):
        return np.random.choice(Pong.NUM_ACTIONS)


class PolicyRight(Policy):
    """An ABC for policies that only implement the right side."""

    def __init__(self, side, name, save_path=None):
        super().__init__(side, name)
    
    def fix_state(self, state):
        if self.side == "l":
            return Pong.flip_state(state)
        else:
            return state
    
    def get_action(self, state, *args):
        return self.get_action_right(self.fix_state(state), *args)

    def get_action_right(self, state, *args):
        raise NotImplementedError()


class TargetPolicy(PolicyRight):
    """An ABC for policies that calculate a target poistion and go there."""

    @classmethod
    def action_for_target(cls, y, target):
        low = y - 0.5 * c.HPL
        high = y + 0.5 * c.HPL
        
        if target < low:
            return c.A_DOWN
        elif target > high:
            return c.A_UP
        else:
            return c.A_STAY

    def __init__(self, side, name, save_path=None):
        super().__init__(side, name)
    
    def get_action_right(self, state, *args):
        target = self.get_target_right(state, *args)
        y = state[S.R_Y]
        return self.action_for_target(y, target)

    def get_target(self, state, *args):
        return self.get_target_right(self.fix_state(state), *args)

    def get_target_right(self, state, *args):
        raise NotImplementedError()


class Follow(TargetPolicy):
    """Follow the Ball"""

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", "-n", default="Follow")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))
    
    def __init__(self, side, name="Follow"):
        super().__init__(side, name)
    
    def get_target_right(self, state, *args):
        return state[S.BALL_Y]


class Predict(TargetPolicy):
    """Predict where the ball will hit and go there"""

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", "-n", default="Predict")
        parser.add_argument("--max_hits", "-m", default=2, type=int)
        parser.add_argument("--epsilon", "-e", default=0.0, type=float)
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, max_hits=2, epsilon=0.0, name="Predict"):
        super().__init__(side, name)
        self.max_hits = max_hits
        self.epsilon = epsilon

    def get_action(self, state, *args):
        if np.random.random() > self.epsilon:
            return self.get_action_right(self.fix_state(state), *args)
        else:
            return np.random.choice(Pong.NUM_ACTIONS)

    @classmethod
    def predict_y(cls, x, state, max_hits):
        time_to_contact = (x - state[S.BALL_X]) / state[S.BALL_VX]

        target = (c.TOP + c.BOTTOM) / 2

        if time_to_contact > 0.0: ## ball is approaching
            ball_y = state[S.BALL_Y]
            ball_vy = state[S.BALL_VY]

            n_hits = 0
            while n_hits <= max_hits:
                next_y = ball_y + ball_vy * time_to_contact
                if next_y < c.BOTTOM: # hit bottom
                    t = (c.BOTTOM - ball_y) / ball_vy
                    ball_y = c.BOTTOM
                    ball_vy = -ball_vy
                    time_to_contact -= t
                    n_hits += 1
                elif next_y > c.TOP: # hit top
                    t = (c.TOP - ball_y) / ball_vy
                    ball_y = c.TOP
                    ball_vy = -ball_vy
                    time_to_contact -= t
                    n_hits += 1
                else:
                    target = next_y
                    break
        return target

    def get_target_right(self, state, *args):
        return self.predict_y(c.RIGHT, state, self.max_hits)


class DiscretePredict(TargetPolicy):
    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", "-n")
        parser.add_argument("--max_hits", "-m", default=2, type=int)
        parser.add_argument("--n_cells", "-nc", default=20, type=int)
        args = parser.parse_args(shlex.split(argv))
        
        if args.name is None:
            args.name = "DiscretePredict({})".format(args.n_cells)
        
        return cls(side, **vars(args))

    def __init__(self, side, max_hits=2, n_cells=20, name="DiscretePredict"):
        super().__init__(side, name)
        self.max_hits = max_hits
        n_bins = [n_cells, n_cells, 3, n_cells,
            n_cells, n_cells, n_cells, n_cells]
        self.d = common.Discretization(Pong.RANGES, n_bins)
        self.new_episode()

    def new_episode(self):
        self.prev_sign = 0

    def get_target_right(self, state, *args):
        state = self.d.discretize(state)
        vys = np.sign(state[S.BALL_VY])
        vxs = np.sign(state[S.BALL_VX])
        if vxs > 0:
            if self.prev_sign == 0:
                self.prev_sign = vys
                self.sign = vys
                self.base = 0
                self.x_path = []
                self.y_path = []
            elif vys != self.prev_sign:
                self.prev_sign = vys
                self.sign *= -1
                self.base += 2
            x = state[S.BALL_X]
            y = self.base + self.sign * state[S.BALL_Y]
            self.x_path.append(x)
            self.y_path.append(y)
            time = np.arange(len(self.y_path))
            A = np.c_[time, np.ones(len(self.y_path))]
            b1, b0 = np.linalg.lstsq(A, self.y_path)[0]
            state[S.BALL_VY] = vys * b1
            y_hat = b0 + b1 * time[-1]
            state[S.BALL_Y] = self.sign * (y_hat - self.base)
            b1, b0 = np.linalg.lstsq(A, self.x_path)[0]
            # state[S.BALL_VX] = b1
            x_hat = b0 + state[S.BALL_VX] * time[-1]
            state[S.BALL_X] = x_hat
            
        else:
            self.prev_sign = 0
        return Predict.predict_y(c.RIGHT, state, self.max_hits)


class Planning(PolicyRight):  
    
    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="Planning")
        parser.add_argument("--random_action", "-ra", action="store_true")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, save_path, name="Planning", random_action=False):
        super().__init__(side, name)
        self.random_action = random_action
        self.VI = ValueIteration(load_path=save_path)
        self.VI.d.set_sphere(1)
    
    def get_action_right_2(self, state, *args):
        if state[S.BALL_VX] < 0:
            target = (c.TOP + c.BOTTOM) / 2
            y = state[S.R_Y]
            return TargetPolicy.action_for_target(y, target)
        
        ind = self.VI.d.state_to_index(state[TRAIN_DIMS])
        next_state = self.VI.next_state[ind]
        if (next_state == -1).any():
            a = self.VI.next_reward[ind].argmax()
        else:
            a = self.VI.V[next_state].argmax()
        return a
    
    
    def get_action_right(self, state, *args):
        if state[S.BALL_VX] < 0:
            target = (c.TOP + c.BOTTOM) / 2
            y = state[S.R_Y]
            return TargetPolicy.action_for_target(y, target)
        
        inds = self.VI.d.state_to_neighborhood(state[TRAIN_DIMS])
        a = np.zeros(Pong.NUM_ACTIONS)
        next_states = self.VI.next_state[inds]
        terminal = (next_states == -1).any(1)
        a = (self.VI.next_reward[next_states[ terminal]].sum(0) +
                       self.VI.V[next_states[~terminal]].sum(0))
        return a.argmax()
    
    def discretization(self, state):
        dstate = self.VI.d.discretize(state[TRAIN_DIMS])
        n_state = state.copy()
        n_state[TRAIN_DIMS] = dstate
        n_state[S.R_VY] = 0.
        return n_state


class NNPolicy(PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="NeuralNetwork")
        parser.add_argument("--n_cells", "-nc", type=int)
        parser.add_argument("--erase_left", "-el", action="store_true")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, n_cells=None, layers=[50, 50, 50], save_path=None,
        erase_left=False, name="NeuralNetwork"):

        super().__init__(side, name)
        self.erase_left = erase_left
        
        if n_cells is not None:
            n_bins = [n_cells, n_cells, 3, n_cells,
                n_cells, n_cells, n_cells, n_cells]
            self.d = common.Discretization(Pong.RANGES, n_bins)
        else:
            self.d = None
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, layers)
            if save_path is not None:
                self.nn.load(save_path)

    def fix_state(self, state):
        state = super().fix_state(state)
        if self.d is not None:
            state = self.d.discretize(state)
        if self.erase_left:
            state = state.copy()
            state[[S.L_Y, S.L_VY]] = 0.0
        return state
    
    def get_action(self, state, *args):
        s = self.fix_state(state)
        return self.nn.predict_argmax(s[None,:])[0]
    
    def prior(self, state):
        s = self.fix_state(state)
        return (self.nn.predict_probabilities(s[None, :])[0], 0.0)

    def prior2(self, state):
        s = self.fix_state(state)
        f = np.vstack([Pong.flip_state(s), s])
        return (self.nn.predict_probabilities(f), 0.0)


class TargetNNPolicy(TargetPolicy):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="NeuralNetwork")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, layers=[50, 50, 50], save_path=None,
        name="NeuralNetwork"):
        
        super().__init__(side, name)
        self.g = tf.Graph()
        with self.g.as_default():
            self.nn = NeuralNetwork(Pong.STATE_DIM, 1, layers)
            if save_path is not None:
                self.nn.load(save_path)

    def get_target_right(self, state, *args):
        return self.nn.predict_raw(state[None,:])[0]


class AlphaPongPolicy(PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="AlphaPong")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, layers=[50, 50, 50], save_path=None,
        name="AlphaPong"):

        super().__init__(side, name)
        self.g = tf.Graph()
        with self.g.as_default():
            self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, layers)
            self.evaluation = tf.tanh(self.nn.affine(
                "evaluation",
                self.nn.layers[-1],
                1,
                relu=False
            )[:, 0])
            if save_path is not None:
                self.nn.load(save_path)
    
    def get_action_right(self, state, *args):
        return self.nn.predict_argmax(state[None,:])[0]
    
    def prior(self, state):
        s = self.fix_state(state)
        feed_dict = {self.nn.input: s[None, :]}
        P, V = self.nn.session.run([self.nn.probabilities, self.evaluation],
            feed_dict)
        return (P[0], V[0])

    def prior2(self, state):
        s = self.fix_state(state)
        feed_dict = {self.nn.input: np.vstack([Pong.flip_state(s), s])}
        P, V = self.nn.session.run([self.nn.probabilities, self.evaluation],
            feed_dict)
        return P, V[1]


class MCTSPolicy(PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", "-n")
        parser.add_argument("--depth", "-md", default=50, type=int)
        parser.add_argument("--c_puct", "-c", default=1.4, type=float)
        parser.add_argument("--num_simulation", "-ns", default=100, type=int)
        parser.add_argument("--base_policy", "-bp")
        parser.add_argument("--base_policy_args", "-bpa", default="")
        parser.add_argument("--self_play", "-sp", action="store_true")
        
        args = parser.parse_args(shlex.split(argv))
        
        if args.base_policy:
            r_pol = POLICIES[args.base_policy].create_from_commandline("r",
                args.base_policy_args)
            if args.name is None:
                args.name = "MCTS({})".format(r_pol.name)
        else:
            r_pol = None
            if args.name is None:
                args.name = "MCTS(U)".format(r_pol.name)
        
        return cls(side, r_pol, **vars(args))

    def __init__(self, side, r_pol=None, name="MCTS", depth=50,
        c_puct=1.4, num_simulations=100, self_play=False, **kwargs):
        super().__init__(side, name)
        self.l_pol = Follow("l")
        
        if r_pol is None:
            uniform = np.ones(Pong.NUM_ACTIONS) / Pong.NUM_ACTIONS
            self.prior = lambda state: (uniform, 0.)
        elif self_play:
            self.prior = r_pol.prior2
        else:
            self.prior = r_pol.prior
        
        self.sim = Pong()
        self.depth = depth
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.self_play = self_play

    def get_action_right(self, state, *args):
        self.sim.set_state(state)
        if self.self_play:
            m = two_sided_mcts.MCTS(self.sim, self.prior,
                self.depth, self.c_puct)
        else:
            m = one_sided_mcts.MCTS(self.sim, self.l_pol, self.prior,
                self.depth, self.c_puct)
        m.search(self.num_simulations)
        p = m.root.probabilities()

        if self.self_play:
            return p[1].argmax()
        else:
            return p.argmax()

        
POLICIES = {
    "manual": Manual,
    "random": RandomPolicy,
    "follow": Follow,
    "predict": Predict,
    "dpredict": DiscretePredict,
    "planning": Planning,
    "nn": NNPolicy,
    "targetnn": TargetNNPolicy,
    "mcts": MCTSPolicy,
    "apz": AlphaPongPolicy
}
