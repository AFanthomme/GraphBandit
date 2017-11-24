""" This script intended to demonstrate the tools from the src package"""
from src import *
# from graph_tool.all import *
import sys
import graph_tool as gt
import graph_tool.topology as tp
import numpy as np


def create_prototype_bandit(n_nodes, plot=False, graph_params=None):
    """
    Draws a random graph

    :param n_nodes:
    :param plot:
    :param seed:
    :return:
    """

    alpha = 0.8

    g = gt.Graph()

    # Add nodes and draw their properties according to graph_params
    ext_flow = g.new_vertex_property('double')
    g.vertex_properties["ext_flow"] = ext_flow
    capacity = g.new_vertex_property('double')
    g.vertex_properties["capacity"] = capacity
    stack = g.new_vertex_property('double')
    g.vertex_properties["stack"] = stack

    for node_index in range(n_nodes):
        v = g.add_vertex()
        g.vertex_properties.ext_flow[v] = np.random.choice([0, 1, -1], p=[0.7, 0.2, 0.1])
        g.vertex_properties.capacity[v] = 5 * np.random.rand() + 1
        g.vertex_properties.stack[v] = 0

    # Add edges and initialize the flow
    edge_flow = g.new_edge_property('double')
    g.edge_properties["flow"] = edge_flow
    edge_resistance = g.new_edge_property('double')
    g.edge_properties["resistance"] = edge_resistance

    for parent in range(n_nodes):
        plop = np.random.randint(n_nodes, size=np.floor(alpha * np.sqrt(n_nodes)).astype(int))
        plop = plop[plop != parent]
        for child in plop:
            e = g.add_edge(g.vertex(parent), g.vertex(child))
            g.edge_properties.flow[e] = 0.
            g.edge_properties.resistance[e] = 2.

    # Remove all cycles from the graph by cutting the first link in the chain.
    # for circuit in tp.all_circuits(g):
    #     # print(circuit)
    #     try:
    #        g.remove_edge(g.edge(circuit[0], circuit[1]))
    #     except:
    #         pass

    g.save('saves/example_{}.gt'.format(n_nodes))

    if plot:
        import graph_tool.draw as gtdraw
        gtdraw.graph_draw(g, vertex_fill_color=g.vertex_properties.ext_flow, vertex_font_size=18,
                              output_size=(600, 600), output="example.png")

u = raw_input('Plot graph (causes crash on exit) (y/n)? ')
create_prototype_bandit(30, u == 'y')
