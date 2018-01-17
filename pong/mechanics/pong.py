from enum import IntEnum

import numpy as np

from . import constants as c


class S(IntEnum):
    BALL_X = 0
    BALL_Y = 1
    BALL_VX = 2
    BALL_VY = 3
    L_Y = 4
    L_VY = 5
    R_Y = 6
    R_VY = 7

    l = 4
    r = 6
    Y = 0
    V = 1


def random_sign():
    return np.random.choice([-1, 1])


class Pong:
    STATE_DIM = 8
    NUM_ACTIONS = 3
    RANGES = [(-1., 1.), (-1., 1.), (-0.0299999, 0.0299999), (-0.2, 0.2),
        (-1., 1.), (-0.2, 0.2), (-1., 1.), (-0.2, 0.2)]
    
    @classmethod
    def empty_state(cls):
        return np.zeros(cls.STATE_DIM, np.float32)

    @classmethod
    def flip_state(cls, state):
        f_state = state.copy()
        f_state[..., [S.BALL_X, S.BALL_VX]] *= -1.
        f_state[..., [S.L_Y, S.L_VY, S.R_Y, S.R_VY]] = \
            state[..., [S.R_Y, S.R_VY, S.L_Y, S.L_VY]]
        return f_state

    def __init__(self, max_steps=c.MAX_STEPS, random_positions=False, f=None):
        """
        Create a new Pong simulator.
        max_steps is the maximum number of steps before a draw. Use
        max_steps=None for limitless games.
        """

        self.reset(random_positions=random_positions)
        if max_steps is None:
            self.max_steps = np.inf
        else:
            self.max_steps = max_steps

        self.f = f

    def set_state(self, state):
        """Set the internal state of the simulator"""

        self.s = state.copy()

        self.done = False
        self.win = None
        self.hit = None
        self.miss = None
        self.n_steps = 0

    def reset(self, random_positions=False):
        """Start the simulator anew"""
        self.s = self.empty_state()
        if random_positions:
            self.s[[S.L_Y, S.R_Y]] = np.random.uniform(
                c.BOTTOM + c.HPL,
                c.TOP - c.HPL,
                2
            )

        self.score = {"l": 0, "r": 0, "draw": 0}
        self.new_episode()

    def new_episode(self):
        """Start a new game episode"""
        s = self.s
        s[S.BALL_X] = 0.
        s[S.BALL_Y] = 0.
        s[S.BALL_VX] = random_sign() * c.VX
        s[S.BALL_VY] = random_sign() * np.random.uniform(c.VY0, c.VY1)

        self.set_state(s)

    def random_state(self):
        """Set the simulator to a random state."""
        s = self.s
        s[S.BALL_VX] = random_sign() * c.VX
        s[[S.BALL_X, S.BALL_Y, S.L_Y, S.R_Y]] = np.random.uniform(-1., 1., 4)
        s[[S.BALL_VY, S.L_VY, S.R_VY]] = np.random.uniform(-0.2, 0.2, 3)
        self.set_state(s)

    def step_paddle(self, p, a):
        """Perform action a on paddle p"""
        s = self.s

        s[p + S.V] = (s[p + S.V] + c.DY[a]) * c.FRICTION
        s[p + S.Y] = s[p + S.Y] + s[p + S.V]

        if s[p + S.Y] + c.HPL >= c.TOP:
            s[p + S.Y] = c.TOP - c.HPL
            s[p + S.V] = 0.0

        if s[p + S.Y] - c.HPL <= c.BOTTOM:
            s[p + S.Y] = c.BOTTOM + c.HPL
            s[p + S.V] = 0.0

    def step_ball(self):
        """
        Overly complicated function to move the ball 1 time unit forward.
        Deals with the rare cases where the ball hits two edges at the same
        time unit.
        """
        s = self.s

        if s[S.BALL_VX] > 0.:
            tt_x = (c.RIGHT - s[S.BALL_X]) / s[S.BALL_VX]
        elif s[S.BALL_VX] < 0.:
            tt_x = (c.LEFT - s[S.BALL_X]) / s[S.BALL_VX]
        else:
            tt_x = np.inf
        
        if s[S.BALL_VY] > 0.:
            tt_y = (c.TOP - s[S.BALL_Y]) / s[S.BALL_VY]
        elif s[S.BALL_VY] < 0.:
            tt_y = (c.BOTTOM - s[S.BALL_Y]) / s[S.BALL_VY]
        else:
            tt_y = np.inf

        if (tt_x > 1.) and (tt_y > 1.): # no collision
            self.advance_ball(1.)

        elif tt_x <= tt_y <= 1.: # collision on X then on Y
            self.advance_ball(tt_x)
            self.hit_x()
            self.advance_ball(tt_y - tt_x)
            self.hit_y()
            self.advance_ball(1. - tt_y)

        elif tt_y < tt_x <= 1.: # collision on Y then on X
            self.advance_ball(tt_y)
            self.hit_y()
            self.advance_ball(tt_x - tt_y)
            self.hit_x()
            self.advance_ball(1. - tt_x)

        elif tt_x <= 1.: # collision on X
            self.advance_ball(tt_x)
            self.hit_x()
            self.advance_ball(1. - tt_x)

        elif tt_y <= 1.: # collision on Y
            self.advance_ball(tt_y)
            self.hit_y()
            self.advance_ball(1. - tt_y)

        else: # ???
            raise RuntimeError("Weird")

    def advance_ball(self, t):
        """
        Move ball t time units, assuming the ball doesn't hit any edge.
        """
        s = self.s
        s[S.BALL_X] += t * s[S.BALL_VX]
        s[S.BALL_Y] += t * s[S.BALL_VY]

    def hit_y(self):
        """Handle the case where the ball hits top or bottom."""
        self.s[S.BALL_VY] *= -1.

    def hit_x(self):
        """Handle the case where the ball hits left or right"""
        s = self.s
        side = "l" if np.isclose(s[S.BALL_X], c.LEFT) else "r"
        p = S[side]

        if s[p + S.Y] - c.HPL < s[S.BALL_Y] < s[p + S.Y] + c.HPL:
            s[S.BALL_VX] *= -1.
            s[S.BALL_VY] += s[p + S.V]
            self.hit = side
        else:
            self.miss = side
            self.win = "l" if (side == "r") else "r"
            self.score[self.win] += 1
            self.done = True

    def step(self, l_a, r_a):
        """
        Advance the simulator 1 time unit forward.
        """

        if self.done:
            raise ValueError("Episode done")

        self.hit = None
        self.miss = None

        self.step_paddle(S.l, l_a)
        self.step_paddle(S.r, r_a)
        
        self.step_ball()

        self.n_steps += 1
        if not self.done and self.n_steps >= self.max_steps:
            self.score["draw"] += 1
            self.done = True
        
        if self.f:
            self.s = self.f(self.s)

    def fast_set_and_step(self, s, l_a, r_a):
        """
        Set the simulator to state s and advance the simulator 1 time unit
        forward.
        """

        self.s[:] = s
        self.hit = None
        self.miss = None
        self.step_paddle(S.l, l_a)
        self.step_paddle(S.r, r_a)
        self.step_ball()

    def get_state(self):
        """
        Get the complete internal state of the game.
        """
        return self.s.copy()

    def reward(self, side, full=True):
        t = self.win if full else self.hit
        if t == side:
            return 1
        elif self.miss == side:
            return -1
        else:
            return 0

    def win_string(self):
        return self.win if self.win else "draw"


def play(sim, l_pol, r_pol, n_episodes = 100):
    """Run n_episodes episodes of Pong using the policies l_pol and r_pol"""

    try:
        for i in range(n_episodes):
            while not sim.done:
                state = sim.get_state()
                l_a = l_pol.get_action(state)
                r_a = r_pol.get_action(state)
                sim.step(l_a, r_a)
            sim.new_episode()
    except KeyboardInterrupt:
        pass
    return sim.score
