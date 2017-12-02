import numpy as np
from scipy.linalg import expm
import pickle


class GraphBandit:
    """
    A GraphBandit is an oriented graph with additional properties.

    """

    def __init__(self, filename='simple'):
        self.graph = np.loadtxt('saves/{}.graph'.format(filename))
        self.ext_flow = np.loadtxt('saves/{}.extflow'.format(filename))
        self.capacity = np.loadtxt('saves/{}.capacities'.format(filename))
        self.nb_nodes = np.shape(self.graph)[0]

        # Start with no stack and no flow
        self.flow = np.zeros((self.nb_nodes, self.nb_nodes))
        self.stack = np.zeros(self.nb_nodes)

        # Start without a priori on the nodes
        self.estimates = np.zeros(self.nb_nodes)

        # Some parameters
        self.time_scale = 0.1

    def evolution_step(self):
        # Use the current value of the flow to diffuse the stack
        d = 1. / np.sqrt(np.sum(self.flow, axis=1))
        laplacian = np.linalg.multi_dot([d, self.flow, d]) - np.eye(self.nb_nodes, self.nb_nodes)

        # Evolve using the exponential of the RW laplacian exp(tL) with a small time-step t.
        self.stack = expm(self.time_scale * laplacian).dot(self.stack)

        # After evolution, get the external in and out flows
        arrivals = np.zeros_like(self.ext_flow)
        for i in range(self.nb_nodes):
            if self.ext_flow[i] > 0:
                arrivals[i] = np.random.poisson(self.ext_flow[i])
            elif self.ext_flow[i] < 0:
                arrivals[i] = self.ext_flow[i]

        self.stack = np.max(np.min(self.stack + arrivals, self.capacity), 0)
        reward = -np.sum(self.stack == self.capacity)    # Could be changed to penalize more large mistakes.
        return reward

    def flow_policy(self):
        # Given the current estimate of the sources and the state of the network, choose the flow to set for the next
        # evolution step.
        pass

if __name__ == '__main__':
    bandit = GraphBandit()