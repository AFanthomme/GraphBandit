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

    def evolution_step(self):
        # Use the current value of the flow to diffuse the stack
        stack = self.network.vertex_properties["stack"]
        capacity = self.network.vertex_properties["capacity"]
        flow = self.network.vertex_properties["flow"].a
        ext_flow = self.network.vertex_properties["ext_flow"].a

        diffusion_kernel = flow # Infinitesimal diffusion step with the current flow
        stack.a = diffusion_kernel.dot(stack.a)

        # After evolution, get the external in and out flows
        arrivals = np.zeros_like(ext_flow)
        for i in range(self.network.num_vertices()):
            if ext_flow[i] > 0:
                arrivals[i] = np.random.poisson(ext_flow[i])
            elif ext_flow[i] < 0:
                arrivals[i] = ext_flow[i]

        stack.a = np.max(np.min(stack.a + arrivals, capacity.a), 0)
        reward = -np.sum(stack.a == capacity.a)    # Could be changed to penalize more large mistakes.
        return reward

    def flow_policy(self):
        # Given the current estimate of the sources and the state of the network, choose the flow to set for the next
        # evolution step.
        pass

