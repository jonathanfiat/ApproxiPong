import numpy as np

from ..mechanics.pong import Pong


class Node:
    def __init__(self, state, P, V, c_puct):
        self.state = state

        self.N = np.zeros(Pong.NUM_ACTIONS, np.int32)
        self.W = np.zeros(Pong.NUM_ACTIONS, np.float32)
        self.Q = np.zeros(Pong.NUM_ACTIONS, np.float32)
        self.child = [None] * Pong.NUM_ACTIONS

        self.P = P
        self.V = V
        self.c_puct = c_puct

    def pick_child(self):
        U = self.c_puct * self.P * np.sqrt(self.N.sum()) / (1. + self.N)
        return (self.Q + U).argmax()

    def propogate_reward(self, a, v):
        self.N[a] += 1
        self.W[a] += v
        self.Q[a] = self.W[a] / self.N[a]

    def probabilities(self, tau=1.):
        if tau == 0.:
            p = np.zeros(Pong.NUM_ACTIONS, np.float32)
            p[self.N.argmax()] = 1.
            return p
        else:
            p = self.N ** (1. / tau)
            return (p / p.sum())


class MCTS:
    def __init__(self, sim, l_pol, prior, max_depth, c_puct):
        self.l_pol = l_pol
        self.sim = sim
        self.prior = prior
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.root = self.create_node()

    def create_node(self):
        state = self.sim.get_state()
        P, V = self.prior(state)
        return Node(state, P, V, self.c_puct)

    def create_child(self, node, a):
        l_a = self.l_pol.get_action(node.state)
        self.sim.set_state(node.state)
        self.sim.step(l_a, a)

        if self.sim.win == "r":
            node.child[a] = 1
        elif self.sim.win == "l":
            node.child[a] = -1
        else:
            node.child[a] = self.create_node()

    def select(self):
        stack = []
        node = self.root

        for i in range(self.max_depth):
            a = node.pick_child()
            stack.append((node, a))

            if node.child[a] is None:
                self.create_child(node, a)

            if node.child[a] in [1, -1]:
                v = node.child[a]
                break

            node = node.child[a]
            v = node.V

        for node, a in stack:
            node.N[a] += 1
            node.W[a] += v
            node.Q[a] = node.W[a] / node.N[a]

    def search(self, num):
        for i in range(num):
            self.select()

    def step(self, a):
        if self.root.child[a] is None:
            self.create_child(self.root, a)

        self.root = self.root.child[a]

    def done(self):
        return self.root in [-1, 1]
