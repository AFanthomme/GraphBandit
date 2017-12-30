import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    shortest = None

    for node in np.where(graph[start, :] > 0)[0]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


class GraphBandit_overstack_deterministic:
    """
    A GraphBandit is an oriented graph with additional properties.

    """

    def __init__(self, filename='example_simple', prefix='../', time_scale=0.1):
        self.graph = np.loadtxt(prefix + 'saves/{}.graph'.format(filename))
        self.ext_flow = np.loadtxt(prefix + 'saves/{}.extflow'.format(filename))
        self.capacity = np.loadtxt(prefix + 'saves/{}.capacities'.format(filename))
        self.nb_nodes = np.shape(self.graph)[0]

        # Start with no stack and flow between all edges
        self.flow = np.ones((self.nb_nodes, self.nb_nodes)) - np.eye(self.nb_nodes, self.nb_nodes)
        self.stack = np.zeros(self.nb_nodes)

        # Start without a priori on the nodes
        self.estimates = np.zeros(self.nb_nodes)

        # Some parameters
        self.time_scale = time_scale

    def UCB(self, n_epochs=100):
        rewards = np.zeros(n_epochs)
        for t in range(n_epochs):
            reward, arrivals = self.evolution_step()
            self.estimates = (t*self.estimates + arrivals) / (t+1)
            self.flow = self.flow_policy()
            rewards[t] = reward
        return rewards

    def evolution_step(self):
        # Use the current value of the flow to diffuse the stack
        d = np.sqrt(np.sum(self.flow, axis=1))

        laplacian = np.array([[self.flow[i, j] / (d[i] * d[j]) for j in range(self.nb_nodes)]
                              for i in range(self.nb_nodes)]) - np.eye(self.nb_nodes, self.nb_nodes)


        # Evolve using the exponential of the RW laplacian exp(tL) with a small time-step t.
        self.stack = expm(self.time_scale * laplacian).dot(self.stack)

        # After evolution, get the external in and out flows
        arrivals = np.zeros_like(self.ext_flow)
        for node in range(self.nb_nodes):
            if self.ext_flow[node] > 0:
                arrivals[node] = np.random.poisson(self.ext_flow[node])
            elif self.ext_flow[node] < 0:
                arrivals[node] = self.ext_flow[node]

        self.stack = np.maximum(np.minimum(self.stack + arrivals, self.capacity), np.zeros_like(arrivals))
        reward = -np.sum(self.stack == self.capacity)    # Could be changed to penalize more large mistakes.
        # print(self.stack)
        return reward, arrivals

    def best_path_to_sink(self, x, sinks_list):
        # Return a list of tuples (parent, child) that lead from x to y
        paths_to_sinks = [find_shortest_path(self.graph, x, y) for y in sinks_list]
        distances = np.array([len(path) if path else 0 for path in paths_to_sinks])
        best_distance = np.min(distances[distances > 0])
        path = paths_to_sinks[np.where(distances == best_distance)[0][0]]

        # For convenience, transform this to a list of tuples
        return [[path[step], path[step+1]] for step in range(len(path)-1)]

    def flow_policy(self, conservativeness=5):
        # Given the current estimate of the sources and the state of the network, choose the flow to set for the next
        # evolution step.
        self.flow = 0.1 * np.ones((self.nb_nodes, self.nb_nodes)) # For regularization
        could_overflow = np.where(self.stack + conservativeness * self.estimates > self.capacity)[0]
        estimated_sinks = np.where(self.estimates <= 0)[0]

        for node in list(could_overflow):
            for edge in self.best_path_to_sink(node, estimated_sinks):
                self.flow[edge[0], edge[1]] += self.stack[node]
        return self.flow


class GraphBandit_overstack:
    """
    A GraphBandit is an oriented graph with additional properties.

    """

    def __init__(self, filename='example_stoch_wells', prefix='../', time_scale=0.1, conservativeness=5):
        self.graph = np.loadtxt(prefix + 'saves/{}.graph'.format(filename))
        self.ext_flow = np.loadtxt(prefix + 'saves/{}.extflow'.format(filename))
        self.capacity = np.loadtxt(prefix + 'saves/{}.capacities'.format(filename))
        self.nb_nodes = np.shape(self.graph)[0]

        # Start with no stack and flow between all edges
        self.flow = np.ones((self.nb_nodes, self.nb_nodes)) - np.eye(self.nb_nodes, self.nb_nodes)
        self.stack = np.zeros(self.nb_nodes)

        # Start without a priori on the nodes
        self.estimates = np.zeros(self.nb_nodes)

        # Some parameters
        self.time_scale = time_scale
        self.conservativeness = conservativeness

    def UCB(self, n_epochs=100):
        rewards = np.zeros(n_epochs)
        estimates = np.zeros(n_epochs)
        for t in range(n_epochs):
            reward, arrivals = self.evolution_step()
            self.estimates = (t*self.estimates + arrivals) / (t+1)
            self.flow = self.flow_policy()
            rewards[t] = reward
            estimates[t] = np.mean((self.estimates - self.ext_flow)**2)
            # print(self.stack, 'stacks')
            # print(rewards[t], 'rewards')
            # print(self.estimates, 'estimates')
        return rewards, estimates

    def evolution_step(self):
        # Use the current value of the flow to diffuse the stack
        d = np.sqrt(np.sum(self.flow, axis=1))

        laplacian = np.array([[self.flow[i, j] / (d[i] * d[j]) for j in range(self.nb_nodes)]
                              for i in range(self.nb_nodes)]) - np.eye(self.nb_nodes, self.nb_nodes)


        # Evolve using the exponential of the RW laplacian exp(tL) with a small time-step t.
        self.stack = expm(self.time_scale * laplacian).dot(self.stack)

        # After evolution, get the external in and out flows
        arrivals = np.zeros_like(self.ext_flow)
        for node in range(self.nb_nodes):
            if self.ext_flow[node] > 0:
                arrivals[node] = np.random.poisson(self.ext_flow[node])
            else:
                arrivals[node] = -np.random.poisson(-self.ext_flow[node])

        self.stack = np.clip(self.stack + arrivals, 0., self.capacity)
        reward = -np.sum(self.stack == self.capacity)    # Could be changed to penalize more large mistakes.

        return reward, arrivals


    def flow_policy(self):
        # Given the current estimate of the sources and the state of the network, choose the flow to set for the next
        # evolution step.
        self.flow = 0.05 * np.ones((self.nb_nodes, self.nb_nodes)) # For regularization
        could_overflow_from_ext = np.where(self.stack + self.conservativeness * self.estimates > self.capacity)[0]
        estimated_sinks = np.where(self.estimates <= 0)[0]

        # Now that not all sinks are equivalent (nor perfect), do not choose shortest path to any sink but path to less
        # risky sink (risk is current stack minus estimated removal plus expected arrival)
        sink_risk = self.stack[estimated_sinks] + self.estimates[estimated_sinks]

        for node in list(could_overflow_from_ext):

            # We have to be careful as there could be no path to the best sink. In that case, we fallback to the next
            # best sink and so on (we only use graphs where at least one sink can be reached, so this loop always
            # terminates).
            not_done = True
            mask = np.zeros_like(sink_risk, dtype=bool)

            while not_done:
                masked_risks = np.ma.array(sink_risk, mask=mask)
                try:
                    best_sink = masked_risks.argmin()
                    path = find_shortest_path(self.graph, node, estimated_sinks[best_sink])
                    tuple_path = [[path[step], path[step + 1]] for step in range(len(path) - 1)]
                except TypeError:
                    mask[best_sink] = True
                    continue
                not_done = False

            for edge in tuple_path:
                self.flow[edge[0], edge[1]] += self.stack[node]
            sink_risk[best_sink] += self.stack[node]

        return self.flow

if __name__ == '__main__':
    n_epochs = 30
    n_repeats = 200
    # time_step_array = [0.25]
    time_step_array = np.linspace(0.1, 0.8, 12)
    mean_perf = np.zeros(len(time_step_array))

    for idx, delta in enumerate(time_step_array):
        print('Time scale #{} = {}'.format(idx, delta), )
        bandit = GraphBandit_overstack(time_scale=delta)
        # bandit = GraphBandit_overstack_deterministic(filename='example_second', time_scale=delta)
        rewards = np.zeros((n_repeats, n_epochs))
        estimates = np.zeros((n_repeats, n_epochs))
        for repeat in range(n_repeats):
            if repeat % 50 == 0:
                print('\tExecuting repeat #{}'.format(repeat))
            rewa, est = bandit.UCB(n_epochs=n_epochs)
            rewards[repeat, :] = rewa
            estimates[repeat, :] = est
        plt.figure(0)
        plt.plot(np.mean(rewards, axis=0), label='scale = {}'.format(np.around(delta, 3)))
        plt.figure(1)
        plt.plot(np.mean(estimates, axis=0), label='scale = {}'.format(np.around(delta, 3)))

        mean_perf[idx] = np.mean(rewards[n_epochs-10:])
    plt.figure(0)
    plt.legend(loc=4)
    plt.figure(1)
    plt.legend(loc=1)

    plt.figure(2)
    plt.plot(time_step_array, mean_perf)


    plt.show()

