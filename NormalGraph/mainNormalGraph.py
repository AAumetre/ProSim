import json
import time

import matplotlib.pyplot as mp
from typing import *
import scipy as scipy
from networkx import DiGraph

from NormalGraph.GraphMonteCarlo import MonteCarlo, NodeType
from NormalMixedLaw import NormalMixedLaw, NormalDensity
from Task import Task
import networkx as nx

Function1D = Callable[[float], float]


def sample_function(f_: Function1D, def_: Tuple[float, float], samples_: int = 1000) -> Tuple[List[float], List[float]]:
    """ Compute the x, y=f(x) arrays, to be used for plotting. """
    min_x, max_x = def_[0], def_[1]
    dx = (max_x - min_x) / samples_
    xs, ys = [0] * samples_, [0] * samples_
    for i in range(samples_):
        xs[i] = min_x + i * dx
        ys[i] = f_(xs[i])
    return xs, ys


def build_integral_function(f_: Function1D, def_: Tuple[float, float]) -> Function1D:
    """ Returns a function f(x) = int(min, x)f(t)dt """
    def integral(x_: float) -> float:
        y, _ = scipy.integrate.quad(f_, def_[0], x_)
        return y
    return integral


def build_mixed_law(graph_, start_) -> NormalMixedLaw:
    """ Uses a NetworkX graph to build a mixed probability distribution law. """
    mixed = NormalMixedLaw(0.01)  # initialise the mixed law
    # need to iterate over nodes and do smth will all edges going out of a node
    densities = extract_densities(graph_, start_, {})
    for density in densities.values():
        mixed.add_normal_densities(density)
    return mixed


def extract_densities(graph_: DiGraph, start_: NodeType, known_densities) -> Dict[Task, List[NormalDensity]]:
    successors = graph_.succ[start_]
    normal_densities = []  # p, m, s
    # at this level, extract the probability densities
    for succ, keys in successors.items():
        p = keys["weight"]
        m, s = succ.normal_
        normal_densities.append((p, m, s))
    known_densities[start_] = normal_densities
    # go to the next level
    for succ, keys in successors.items():
        if succ not in known_densities:  # avoid re-visit
            known_densities.update(extract_densities(graph_, succ, known_densities))
    return known_densities

def compute_esperance(f_: Function1D, def_: Tuple[float, float]) -> float:
    """ Computes the esperance of X, where f_ is the probability density function. """
    e, _ = scipy.integrate.quad(lambda x: x*f_(x), *def_)
    return e


def compute_variance(f_: Function1D, esp_: float, def_: Tuple[float, float]) -> float:
    """ Computes the variance (sigma-squared) of X, where f_ is the probability density function. """
    v, _ = scipy.integrate.quad(lambda x: f_(x)*(x-esp_)**2, *def_)
    return v


def load_graph_json(path_: str):
    """ Load TasksDefinition.json and FlowDefinition.json and return all the necessary objets. """
    starting_node, nodes = None, {}
    # read tasks definitions
    with open(path_+"/TasksDefinition.json") as f:
        data = json.load(f)
        for t in data["tasks"]:
            nodes[t["name"]] = Task(t["name"], (t["normal"][0], t["normal"][1]), t["layer"])
        starting_node = nodes[data["starting_task"]]
    edges = []
    # read edges definition
    with open(path_+"/FlowDefinition.json") as f:
        data = json.load(f)
        for e in data["weighted_edges"]:
            edges.append((nodes[e["from"]], nodes[e["to"]], e["p"]))
    # create the graph
    graph = nx.DiGraph()
    for name, node in nodes.items():
        graph.add_node(node, layer=node.layer_)
    graph.add_weighted_edges_from(edges)
    return starting_node, graph


def main():
    # TODO: in JSON files, have a different way to define sigma, other than giving its value
    # TODO: check out Markov chains, might be a bit restrictive
    # TODO: fix mixed law (convolution?)
    # TODO: set-up automatic testing
    # TODO: we might have to set requirements on input graphs topology (and automatically fix it later?)
    # TODO: have the NormalMixedLaw compute its definition set, based on sigmas
    mp.style.use("bmh")

    # Define a graph describing the process, starting with the tasks
    start_node, g = load_graph_json("Examples/SimpleTwoRiskProcess")

    # do a Monte-Carlo simulation
    solver = MonteCarlo(g, start_node)
    mc_sample = solver.compute_sample(1000, True)

    # draw the graph
    options = {"node_size": 3000, "font_color": "white", "arrowsize": 20}
    positioning = nx.multipartite_layout(g, subset_key="layer")
    nx.draw_networkx(g, pos=positioning, with_labels=True, font_weight='bold', **options)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, positioning, edge_labels)  # shown later

    # browse the graph and build the mixed probability law
    mixed = build_mixed_law(g, start_node)
    fig, ax = mp.subplots()
    ax.hist(mc_sample, bins=100)
    mixed.phi_ = [y/0.8*40 for y in mixed.phi_]
    sampled_mixed = mixed.get_sampling()
    ax.plot(*sampled_mixed)
    mp.show()
    return



    mixed_phi = mixed.get_function()
    integral, error = scipy.integrate.quad(mixed_phi, *definition)
    mass_f = build_integral_function(lambda x: mixed_phi(x) / integral, definition)

    # plot density and mass functions
    fig, ax1 = mp.subplots()
    color = 'tab:green'
    ax1.set_xlabel("Duration [h]")
    ax1.set_ylabel("Density function", color=color)
    ax1.plot(*sample_function(lambda x: 420*mixed_phi(x)/6.5, definition, samples_=1000), color=color)
    ax1.hist(mc_sample, bins=100)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Mass function", color=color)
    ax2.plot(*sample_function(mass_f, definition, samples_=1000), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    mp.title("Process duration probability")
    ax1.grid()
    ax2.grid()
    mp.show()

    expected = compute_esperance(mixed_phi, definition)/integral
    print(f"Esperance: {expected:.2f} hours.")
    variance = compute_variance(mixed_phi, expected, definition)/integral
    print(f"Variance: {variance:.2f}.")
    ninety = scipy.optimize.fsolve(lambda x: mass_f(x) - 0.9, [expected])  # start at the esperance value
    print(f"Maximum duration, with 90% confidence: {ninety[0]:.2f} hours.")


if __name__ == '__main__':
    start_time = time.time_ns()
    main()
    print(f"Duration: {(time.time_ns() - start_time) / 10 ** 9} s")
