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
        capacity[node_index] = 100
        if node_index in [0, 1, 3]:
            ext_flow[node_index] = 1
        elif node_index == 4:
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


def make_second_example():
    n_nodes = 10
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    capacity = np.zeros(n_nodes)
    ext_flow = np.zeros(n_nodes)

    for node_index in range(n_nodes):
        capacity[node_index] = 20
        if node_index not in [4, 9, 10]:
            ext_flow[node_index] = 1
        elif node_index in [4, 9]:
            ext_flow[node_index] = -3

    edges = [
        [0, 1],
        [0, 2],
        [1, 4],
        [3, 4],
        [2, 5],
        [5, 3],
        [4, 6],
        [6, 7],
        [6, 8],
        [8, 7],
        [7, 9],
    ]

    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1

    np.savetxt('saves/example_{}.graph'.format('second'), adjacency_matrix)
    np.savetxt('saves/example_{}.extflow'.format('second'), ext_flow)
    np.savetxt('saves/example_{}.capacities'.format('second'), capacity)

def make_stochastic_wells_example():
    n_nodes = 10
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    capacity = np.zeros(n_nodes)
    ext_flow = np.zeros(n_nodes)

    for node_index in range(n_nodes):
        capacity[node_index] = 10
        if node_index not in [4, 9, 3]:
            ext_flow[node_index] = np.random.uniform(0.5, 3)
        elif node_index in [4, 9]:
            ext_flow[node_index] = np.random.uniform(-6, -0.5)

    edges = [
        [0, 1],
        [0, 2],
        [1, 4],
        [3, 4],
        [2, 5],
        [5, 3],
        [4, 6],
        [6, 7],
        [6, 8],
        [8, 7],
        [7, 9],
    ]

    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1

    np.savetxt('saves/example_{}.graph'.format('stoch_wells'), adjacency_matrix)
    np.savetxt('saves/example_{}.extflow'.format('stoch_wells'), ext_flow)
    np.savetxt('saves/example_{}.capacities'.format('stoch_wells'), capacity)


if __name__ == '__main__':
    make_simple_example()
    make_second_example()
    make_stochastic_wells_example()


