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


class AlphaPongPolicy(policies.PolicyRight):

    @classmethod
    def create_from_commandline(cls, side, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("save_path")
        parser.add_argument("--name", "-n", default="NeuralNetwork")
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

    def prior(self, state, *args):
        a = gl.getcurrent().parent.switch(state)
        return a[:-1], a[-1]
    
    def prior2(self, state, *args):
        a_l = gl.getcurrent().parent.switch(Pong.flip_state(state))
        a_r = gl.getcurrent().parent.switch(state)
        return np.vstack((a_l[:-1], a_r[:-1])), a_r[-1]
    
    def evaluate_all(self, states):
        feed_dict = {self.nn.input: states}
        P, V = self.nn.session.run([self.nn.probabilities, self.evaluation],
            feed_dict)
        return np.hstack([P, V[:, None]])


class FastEvaluate:
    def __init__(self, save_path, self_play=False):
        self.l_pol = policies.Follow("l")
        self.save_path = save_path
        self.self_play = self_play
        self.base_policy = AlphaPongPolicy("r", save_path=self.save_path)

    def run_episode(self, *args):
        sim = Pong(random_positions=True)
        r_pol = policies.MCTSPolicy("r", self.base_policy,
            self_play=self.self_play)

        while not sim.done:
            state = sim.get_state()
            l_a = self.l_pol.get_action(state)
            r_a = r_pol.get_action(state)
            sim.step(l_a, r_a)

        return sim.win_string()

    def run_episodes(self, n):
        eval_states = np.zeros((n, Pong.STATE_DIM), np.float32)
        
        threads = []
        for i in range(n):
            t = gl.greenlet(self.run_episode)
            threads.append(t)
        
        A = np.zeros((n, Pong.NUM_ACTIONS + 1))
        alive = np.ones(n, np.bool)

        while alive.any():
            flags = np.zeros(n, np.bool)
            
            for i in range(n):
                if alive[i]:
                    data = threads[i].switch(A[i])
                    if not threads[i].dead:
                        eval_states[i] = data
                        flags[i] = True
                    else:
                        alive[i] = False
                        self.score[data] += 1
            A[flags] = self.base_policy.evaluate_all(eval_states[flags])

    def estimate(self, n):
        self.score = {"l": 0, "r": 0, "draw": 0}
        self.run_episodes(n)
        return self.score


policies.POLICIES["apz"] = AlphaPongPolicy
