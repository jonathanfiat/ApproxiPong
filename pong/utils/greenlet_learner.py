from collections import namedtuple

import greenlet as gl
import numpy as np

from ..mechanics.pong import Pong
from ..mechanics  import constants as c
from ..utils import common


EpisodeResult = namedtuple("EpisodeResult", ["states", "actions", "rewards",
    "winner"])


class GreenletLearner:
    """A base class for learners using the Greenlt library.
    Handles the efficient running of many episodes "in parrallel" and uniting
    the calls to the decision mechanism (in cases like ANN where each call
    is heavy, but uniting many decision in one call is not.
    
    A subclass should implement:
        1. self.iteration(results): do whatever with the episodes results (train
            NN, update db, etc.).
        2. self.predict_actions(eval_states): make a decision for all states
            in eval_states.
    """

    def __init__(self, args):
        """Create the learner using args."""

        if args.self:
            self.l_pol = None
        else:
            self.l_pol = common.read_policy_args(args, "l")
        self.save_path, self.load_path = common.read_save_load_args(args)

        self.args = args
        self.main = gl.getcurrent()
        self.n_iter = 0
        self.score = {"l": 0, "r": 0, "draw": 0}

    def run_episode_opponent(self, *args):
        """Simulate a single episode against a predefined policy."""

        sim = Pong(random_positions=True)
        
        states = np.zeros((1, c.MAX_STEPS, Pong.STATE_DIM), np.float32)
        actions = np.zeros((1, c.MAX_STEPS), np.int32)
        rewards = np.zeros((1, c.MAX_STEPS), np.float32)

        t = 0
        while not sim.done:
            state = sim.get_state()
            l_a = self.l_pol.get_action(state)
            r_a = self.main.switch(state)
            sim.step(l_a, r_a)
            states[0, t] = state
            actions[0, t] = r_a
            rewards[0, t] = sim.reward("r", not self.args.partial)
            t += 1

        return EpisodeResult(states[:, :t], actions[:, :t], rewards[:, :t],
            sim.win_string())

    def run_episode_self(self, *args):
        """Simulate a single episode in a self-play scenario."""

        sim = Pong(random_positions=True)
        
        states = np.zeros((2, c.MAX_STEPS, Pong.STATE_DIM), np.float32)
        actions = np.zeros((2, c.MAX_STEPS), np.int32)
        rewards = np.zeros((2, c.MAX_STEPS), np.float32)

        t = 0
        while not sim.done:
            r_state = sim.get_state()
            l_state = Pong.flip_state(r_state)
            r_a = self.main.switch(r_state)
            l_a = self.main.switch(l_state)
            sim.step(l_a, r_a)
            states[0, t] = r_state
            actions[0, t] = r_a
            rewards[0, t] = sim.reward("r", not self.args.partial)
            states[1, t] = l_state
            actions[1, t] = l_a
            rewards[1, t] = sim.reward("l", not self.args.partial)
            t += 1

        return EpisodeResult(states[:, :t], actions[:, :t], rewards[:, :t],
            sim.win_string())

    def run_episodes(self):
        """Simulate many episode "in parrallel" using greenlet."""

        results = []
        eval_states = np.zeros((self.args.episodes_per_iteration,
            Pong.STATE_DIM), np.float32)

        threads = []
        alive = []
        ## Initialize pseudo-threads
        for i in range(self.args.episodes_per_iteration):
            if self.l_pol is None:
                t = gl.greenlet(self.run_episode_self)
            else:
                t = gl.greenlet(self.run_episode_opponent)
            t.index = i
            threads.append(t)
            alive.append(i)

        A = [None] * self.args.episodes_per_iteration

        while threads:
            count = 0
            for t, a in zip(threads[:], A):
                data = t.switch(a)
                if not t.dead:
                    eval_states[count] = data
                    count += 1
                else:
                    threads.remove(t)
                    alive.remove(t.index)
                    self.score[data.winner] += 1
                    results.append(data)

            A = self.decide_actions(eval_states[:count], alive)

        return results

    def learn(self):
        """Repeat: 1. Episodes simulation. 2. Learning iteration."""

        for self.n_iter in range(self.n_iter, self.args.n_iters):
            results = self.run_episodes()
            data = self.iteration(results)

            if (self.n_iter + 1) % self.args.save_frequency == 0:
                p = self.nn.save(self.save_path, self.n_iter + 1)
                if data is not None:
                    np.savez_compressed(p / "data.npz", **data)

                print("{:04d} : {l}|{draw}|{r}".format(self.n_iter + 1,
                    **self.score))
                self.score = {"l": 0, "r": 0, "draw": 0}

    def iteration(self, results):
        raise NotImplementedError()

    def decide_actions(self, eval_states, alive):
        """
        eval_states: an array of states for decision-making.
        alive: a list on indeces, if the decision process is stateful.
        """

        raise NotImplementedError()
