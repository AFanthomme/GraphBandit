"""
This module can be used to create graphs and save their description in the designated location.
"""
import numpy as np


def make_simple_example():
    n_nodes = 5
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    capacity = np.zeros(n_nodes)
    ext_flow = np.zeros(n_nodes)

    for node_index in range(n_nodes):
        capacity[node_index] = 2 * np.random.rand() + 1
        if node_index in [0, 1, 4]:
            ext_flow[node_index] = 1
        elif node_index == 5:
            ext_flow[node_index] = -100

    edges = [
        [0, 1],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 4],
    ]

    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1

    np.savetxt('saves/example_{}.graph'.format('simple'), adjacency_matrix)
    np.savetxt('saves/example_{}.extflow'.format('simple'), ext_flow)
    np.savetxt('saves/example_{}.capacities'.format('simple'), capacity)


if __name__ == '__main__':
    make_simple_example()


