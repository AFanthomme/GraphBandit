import graph_tool as import gt
import numpy as np


class GraphBandit:
    """
    A GraphBandit is an oriented graph
    """

    def __init__(self, filename):
        self.network = load_graph('saves/{}.gt'.format(filename))
        self.sources = GraphView(self.network, vfilt=lambda v: v.ext_flow != 0).vertices()
        self.state = None

    def get_extflow(self):
        # Get the Poisson-distributed external flow
        # If any node goes overstack, negative reward
        # If reach a terminal state, positive reward
        stack = self.network.vertex_properties["stack"]
        capacity = self.network.vertex_properties["capacity"]
        ext_flow = self.network.vertex_properties["ext_flow"].a
        arrivals = np.zeros_like(ext_flow)
        for i in range(self.network.num_vertices()):
            if ext_flow[i] > 0:
                arrivals[i] = np.random.poisson(ext_flow[i])

        stack.a = np.min(stack.a + arrivals, capacity.a)
        reward = -np.sum(stack.a == capacity.a)


    def diffuse_state(self):
        # Use the chosen value of the flow to evolve the state
        stack = self.network.vertex_properties["stack"]
        capacity = self.network.vertex_properties["capacity"]
        flow = self.network.vertex_properties["flow"]

        stack.a = np.min(flow.a.dot(stack.a), capacity.a)
        reward = -np.sum(stack.a == capacity.a)
        pass

    def flow_policy(self):
        # Given the current estimate of the sources and the state of the network,
        # choose the flow to set
        pass

