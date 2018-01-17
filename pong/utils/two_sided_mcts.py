import numpy as np

from ..mechanics.pong import Pong


class Node:
    def __init__(self, state, P, V, c_puct):
        self.state = state

        self.N = np.zeros((2, Pong.NUM_ACTIONS), np.int32)
        self.W = np.zeros((2, Pong.NUM_ACTIONS), np.float32)
        self.Q = np.zeros((2, Pong.NUM_ACTIONS), np.float32)
        self.child = np.zeros((Pong.NUM_ACTIONS, Pong.NUM_ACTIONS), np.object)

        self.P = P
        self.V = V
        self.c_puct = c_puct

    def pick_child(self):
        U = self.c_puct * self.P * np.sqrt(self.N.sum(1, keepdims=True)) / (1. + self.N)
        return (self.Q + U).argmax(1)

    def propogate_reward(self, l_a, r_a, v):
        ind = ([0, 1], [l_a, r_a])
        self.N[ind] += 1
        self.W[ind] += [-v, v]
        self.Q[ind] = self.W[ind] / self.N[ind]

    def probabilities(self, tau=1.):
        if tau == 0.:
            p = np.zeros((2, Pong.NUM_ACTIONS), np.float32)
            p[[0,1],self.N.argmax(1)] = 1.
            return p
        else:
            p = self.N ** (1. / tau)
            return p / p.sum(1, keepdims=True)


class MCTS:
    def __init__(self, sim, prior, max_depth, c_puct):
        self.sim = sim
        self.prior = prior
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.root = self.create_node()

    def create_node(self):
        state = self.sim.get_state()
        P, V = self.prior(state)
        return Node(state, P, V, self.c_puct)

    def create_child(self, node, l_a, r_a):
        self.sim.set_state(node.state)
        self.sim.step(l_a, r_a)

        if self.sim.win == "r":
            node.child[l_a, r_a] = 1
        elif self.sim.win == "l":
            node.child[l_a, r_a] = -1
        else:
            node.child[l_a, r_a] = self.create_node()

    def select(self):
        stack = []
        node = self.root

        for i in range(self.max_depth):
            l_a, r_a = node.pick_child()
            stack.append((node, l_a, r_a))

            if node.child[l_a, r_a] == 0:
                self.create_child(node, l_a, r_a)

            if node.child[l_a, r_a] in [1, -1]:
                v = node.child[l_a, r_a]
                break

            node = node.child[l_a, r_a]
            v = node.V

        for node, l_a, r_a in stack:
            node.propogate_reward(l_a, r_a, v)

    def search(self, num):
        for i in range(num):
            self.select()

    def step(self, l_a, r_a):
        if self.root.child[l_a, r_a] == 0:
            self.create_child(self.root, l_a, r_a)

        self.root = self.root.child[l_a, r_a]

    def done(self):
        return self.root in [-1, 1]
