import copy
import itertools
import json
import time

import matplotlib.pyplot as mp
from typing import *
import scipy as scipy
from networkx import DiGraph

from NormalGraph.GraphMonteCarlo import MonteCarlo, NodeType
from NormalMixedLaw import NormalMixedLaw
from Task import Task
import networkx as nx

Function1D = Callable[[float], float]
DiscreteFunction1D = Tuple[List[float], List[float]]


def build_mixed_law(graph_: DiGraph, start_: NodeType, end_: NodeType) -> DiscreteFunction1D:
    """ Uses a NetworkX graph to build a mixed probability distribution law. """
    laws = []
    # explore the graph and build a law on each path
    paths = nx.all_simple_paths(graph_, start_, end_)
    for path in paths:
        mixed = NormalMixedLaw(0.01)  # initialise the mixed law
        law_p = 1.0
        for u, v in itertools.pairwise(path):
            p = graph_[u][v]["weight"]
            m, s = v.normal_
            mixed.add_normal_densities([(p, m, s)])
            law_p *= p
        laws.append((law_p, mixed))
    # sum all the laws together
    mixed_x, mixed_y = laws[0][1].get_sampling()  # starts at zero
    mixed_y = [y * laws[0][0] for y in mixed_y]
    dx = laws[0][1].phi_dx_
    for law in laws[1:]:  # TODO: can be optimized
        other_x, other_y = law[1].get_sampling()
        other_y = [y * law[0] for y in other_y]
        mixed_x, mixed_y = sum_discrete_functions((mixed_x, mixed_y), (other_x, other_y), dx)
    return mixed_x, mixed_y


def sum_discrete_functions(f1_: DiscreteFunction1D, f2_: DiscreteFunction1D, dx_: float) -> DiscreteFunction1D:
    # build new range that covers the whole definition set
    in_f1 = lambda n: (f1_[0][0] <= n <= f1_[0][-1])
    in_f2 = lambda n: (f2_[0][0] <= n <= f2_[0][-1])
    min_x, max_x = min(f1_[0][0], f2_[0][0]), max(f1_[0][-1], f2_[0][-1])
    offset_f1 = (int((f1_[0][0] - min_x) / dx_) + 1 if f1_[0][0] > min_x else 0)
    offset_f2 = (int((f2_[0][0] - min_x) / dx_) + 1 if f2_[0][0] > min_x else 0)
    new_xs = [min_x + i * dx_ for i in range(int((max_x - min_x) / dx_))]
    new_ys = [0] * len(new_xs)
    for i, x in enumerate(new_xs):
        if in_f1(x):
            id_f1 = i - offset_f1
            new_ys[i] += f1_[1][id_f1]
        if in_f2(x):
            id_f2 = i - offset_f2
            new_ys[i] += f2_[1][id_f2]
    return new_xs, new_ys


def compute_esperance(f_: DiscreteFunction1D) -> float:
    """ Computes the esperance of X, where f_ is the probability density function. """
    fv = copy.deepcopy(f_[1])
    for i, v in enumerate(fv):
        fv[i] = v*f_[0][i]
    integral = scipy.integrate.cumulative_trapezoid(fv, f_[0])
    return max(integral)


def compute_variance(f_: DiscreteFunction1D, esp_: float) -> float:
    """ Computes the variance (sigma-squared) of X, where f_ is the probability density function. """
    fv = copy.deepcopy(f_[1])
    for i, v in enumerate(fv):
        fv[i] = v*(f_[0][i] - esp_)**2
    integral = scipy.integrate.cumulative_trapezoid(fv, f_[0])
    return max(integral)


def load_graph_json(path_: str):
    """ Load RiskAlternative.json and FlowDefinition.json and return all the necessary objets. """
    starting_node, ending_node, nodes, edges = None, None, {}, []
    # read tasks definitions
    with open(path_) as f:
        data = json.load(f)
        for t in data["tasks"]:
            nodes[t["name"]] = Task(t["name"], (t["normal"][0], t["normal"][1]), t["layer"])
        starting_node = nodes[data["starting_task"]]
        ending_node = nodes[data["ending_task"]]
        # read edges definition
        for e in data["weighted_edges"]:
            edges.append((nodes[e["from"]], nodes[e["to"]], e["p"]))
    # create the graph
    graph = nx.DiGraph()
    for name, node in nodes.items():
        graph.add_node(node, layer=node.layer_)
    graph.add_weighted_edges_from(edges)
    return starting_node, ending_node, graph


def main():
    # TODO: in JSON files, have a different way to define sigma, other than giving its value
    # TODO: check out Markov chains, might be a bit restrictive
    # TODO: set-up automatic testing
    # TODO: we might have to set requirements on input graphs topology (and automatically fix it later?)
    mp.style.use("bmh")

    # Define a graph describing the process, starting with the tasks
    start_node, end_node, g = load_graph_json("Examples/LowRisk.json")

    # do a Monte-Carlo simulation
    solver = MonteCarlo(g, start_node)
    mc_sample = solver.compute_sample(10000, True)

    # draw the graph
    options = {"node_size": 3000, "font_color": "white", "arrowsize": 20}
    positioning = nx.multipartite_layout(g, subset_key="layer")
    nx.draw_networkx(g, pos=positioning, with_labels=True, font_weight='bold', **options)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, positioning, edge_labels)  # shown later

    # browse the graph and build the mixed probability law
    mixed = build_mixed_law(g, start_node, end_node)
    mass_function = scipy.integrate.cumulative_trapezoid(mixed[1], mixed[0])
    mixed = mixed[0], [320 * y for y in mixed[1]]  # scale up

    # plot density and mass functions
    fig, ax1 = mp.subplots()
    color = 'tab:green'
    ax1.set_xlabel("Cost [h]")
    ax1.set_ylabel("Density function", color=color)
    ax1.hist(mc_sample, bins=100)
    ax1.plot(mixed[0], mixed[1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Mass function", color=color)
    ax2.plot(mixed[0][:-1], mass_function, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    mp.title("Process cost probability")
    ax1.grid()
    ax2.grid()
    mp.show()

    # print out some metadata
    integral = sum(mixed[1])*0.01
    expected = compute_esperance(mixed) / integral
    print(f"Esperance: {expected:.2f} hours.")
    variance = compute_variance(mixed, expected) / integral
    print(f"Variance: {variance:.2f}.")


if __name__ == '__main__':
    start_time = time.time_ns()
    main()
    print(f"Duration: {(time.time_ns() - start_time) / 10 ** 9} s")
