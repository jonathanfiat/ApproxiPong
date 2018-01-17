import argparse
import shlex

import greenlet as gl
import numpy as np
import tensorflow as tf

from pong.mechanics.pong import Pong, S
from pong.mechanics import policies
from pong.mechanics import constants as c
from pong.utils import common
from pong.utils.tf_machinery import NeuralNetwork


class NNPolicy(policies.PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="NeuralNetwork")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, layers=[50, 50, 50], save_path=None,
        name="Neural"):
        super().__init__(side, name)
        self.save_path = save_path
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.nn = NeuralNetwork(Pong.STATE_DIM, Pong.NUM_ACTIONS, layers)
            if save_path is not None:
                self.nn.load(save_path)

    def get_action_right(self, state, *args):
        return gl.getcurrent().parent.switch((self.side, state))
    
    def evaluate_all(self, states):
        return self.nn.predict_argmax(states)


class TargetNNPolicy(policies.PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="NeuralNetwork")
        args = parser.parse_args(shlex.split(argv))
        return cls(side, **vars(args))

    def __init__(self, side, layers=[50, 50, 50], save_path=None, name="Neural"):
        super().__init__(side, name)
        self.save_path = save_path
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.nn = NeuralNetwork(Pong.STATE_DIM, 1, layers)
            if save_path is not None:
                self.nn.load(save_path)
    
    def get_action_right(self, state, *args):
        return gl.getcurrent().parent.switch((self.side, state))
    
    def evaluate_all(self, states):
        y = states[:, S.R_Y]
        low = y - 0.5 * c.HPL
        high = y + 0.5 * c.HPL
        
        T = self.nn.predict_raw(states)[:, 0]
        A = np.zeros(states.shape[0], np.int32)
        A[T < low] = c.A_DOWN
        A[T > high] = c.A_UP
        return A


class FastEvaluate:
    def __init__(self, l_pol, r_pol, disc=False):
        self.l_pol = l_pol
        self.r_pol = r_pol
        self.disc = disc

    def run_episode(self, *args):
        if self.disc:
            sim = Pong(random_positions=True, f=self.r_pol.discretization)
        else:
            sim = Pong(random_positions=True)

        while not sim.done:
            state = sim.get_state()
            l_a = self.l_pol.get_action(state)
            r_a = self.r_pol.get_action(state)
            sim.step(l_a, r_a)

        return sim.win_string()

    def run_episodes(self, n):
        eval_states = np.zeros((n, Pong.STATE_DIM), np.float32)
        
        threads = []
        for i in range(n):
            t = gl.greenlet(self.run_episode)
            threads.append(t)
        
        A = [None] * n
        alive = np.ones(n, np.bool)

        while alive.any():
            flags = {"l": np.zeros(n, np.bool), "r": np.zeros(n, np.bool)}
            
            for i in range(n):
                if alive[i]:
                    data = threads[i].switch(A[i])
                    if not threads[i].dead:
                        side, state = data
                        eval_states[i] = state
                        flags[side][i] = True
                    else:
                        alive[i] = False
                        self.score[data] += 1
            
            A = np.zeros(n, np.int32)
            if flags["l"].any():
                A[flags["l"]] = self.l_pol.evaluate_all(eval_states[flags["l"]])
            if flags["r"].any():
                A[flags["r"]] = self.r_pol.evaluate_all(eval_states[flags["r"]])

    def estimate(self, n):
        self.score = {"l": 0, "r": 0, "draw": 0}
        self.run_episodes(n)
        return self.score


policies.POLICIES["nn"] = NNPolicy
policies.POLICIES["targetnn"] = TargetNNPolicy
