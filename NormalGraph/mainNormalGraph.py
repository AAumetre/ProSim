import copy
import itertools
import json
import time

import matplotlib.pyplot as mp
from typing import *

import numpy
import scipy as scipy
from networkx import DiGraph

from NormalGraph.GraphMonteCarlo import MonteCarlo, NodeType
from Task import Task
import networkx as nx

Function1D = Callable[[float], float]
DiscreteFunction1D = Tuple[List[float], List[float]]


def sum_discrete_functions(f1_: DiscreteFunction1D, f2_: DiscreteFunction1D, dx_: float) -> DiscreteFunction1D:
    # build new range that covers the whole definition set
    def in_f1(n): return f1_[0][0] <= n <= f1_[0][-1]
    def in_f2(n): return f2_[0][0] <= n <= f2_[0][-1]
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


def compute_expectation(f_: DiscreteFunction1D) -> float:
    """ Computes the expectation of X, where f_ is the probability density function. """
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
            # two possibilities to evaluate the distribution: either "three-points" or "normal"
            # in both cases, we need to evaluation the expectation and variance
            if "normal" in t:
                expectation = t["normal"][0]
                variance = t["normal"][1]
            elif "three-points" in t:
                tp = sorted(t["three-points"])
                expectation = (tp[0] + 4*tp[1] + tp[2])/6.0
                variance = max((tp[2] - tp[0]) / 6.0, 1e-6)
            else:
                print(f"Invalid line in file, cannot process task {t}")
            nodes[t["name"]] = Task(t["name"], (expectation, variance), t["layer"])
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
    # TODO: in JSON files, have a different way to define sigma, other than giving its value, use this:
    #       https://en.wikipedia.org/wiki/Three-point_estimation
    # TODO: handle zero-variance durations better than having s->1/inf (which causes discretization issues)
    # TODO: check out Markov chains, might be a bit restrictive
    # TODO: set-up automatic testing
    # TODO: we might have to set requirements on input graphs topology (and automatically fix it later?)
    # TODO: distributions with a mean close to 0.0 (<1.0) seem to cause discrepancies. It looks like the negative
    #       durations - albeit not making much sense - are not taken into account in the mixed law.
    mp.style.use("bmh")

    # Define a graph describing the process, starting with the tasks
    start_node, end_node, g = load_graph_json("Examples/SimpleRiskThreePoints.json")

    # do a Monte-Carlo simulation
    solver = MonteCarlo(g, start_node)
    n_mc_samples = 10_000
    mc_sample = solver.compute_sample(n_mc_samples, True)

    # draw the graph
    options = {"node_size": 3000, "font_color": "black", "arrowsize": 20}
    positioning = nx.multipartite_layout(g, subset_key="layer")
    nx.draw_networkx(g, pos=positioning, with_labels=True, font_weight='bold', **options)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, positioning, edge_labels)  # shown later

    dx = 0.01

    # plot density and mass functions
    fig, ax1 = mp.subplots()
    color = 'tab:green'
    ax1.set_xlabel("Cost [h]")
    ax1.set_ylabel("Density function", color=color)
    n_bins, mc_intervals, _2 = ax1.hist(mc_sample, bins=n_mc_samples//20)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Mass function", color=color)
    mass_carlo = scipy.integrate.cumulative_trapezoid(n_bins, mc_intervals[:-1])
    mass_carlo = [v/max(mass_carlo) for v in mass_carlo]  # normalize mass function
    ax2.plot(mc_intervals[:-2], mass_carlo, "--")
    ax2.tick_params(axis='y', labelcolor=color)
    mp.title("Process cost probability")
    ax1.grid()
    ax2.grid()
    mp.show()

    # print out some metadata
    expected = numpy.mean(mc_sample)
    print(f"Expectation: {expected:.2f} hours.")
    variance = numpy.var(mc_sample)
    print(f"Variance: {variance:.2f}.")


if __name__ == '__main__':
    start_time = time.time_ns()
    main()
    print(f"Duration: {(time.time_ns() - start_time) / 10 ** 9} s")
