import json

import matplotlib.pyplot as mp
from typing import *
import scipy as scipy

from NormalMixedLaw import NormalMixedLaw
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
    mixed = NormalMixedLaw({}, (0.0, 0.1))  # initialise the mixed law
    bfs_edges = nx.bfs_edges(graph_, start_)
    for u, v in bfs_edges:
        p = graph_[u][v]["weight"]
        m, s = v.normal_
        if m != 0.0 and s != 0.0:
            mixed.add_normal((p, *v.normal_))
    return mixed


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
    for node in nodes.values():
        graph.add_node(node, layer=node.layer_)
    graph.add_weighted_edges_from(edges)
    return starting_node, graph


def main():
    # TODO: in JSON files, have a different way to define sigma, other than giving its value
    # TODO: implement Monte-Carlo simulation, to cross-check the computation of mixed law (currently wrong)
    # TODO: fix mixed law (convolution?)
    # TODO: we might have to set requirements on input graphs topology (and automatically fix it later?)
    # TODO: have the NormalMixedLaw compute its definition set, based on sigmas

    # Define a graph describing the process, starting with the tasks
    start_node, g = load_graph_json("Examples/LowRiskStraightProcess")

    # draw the graph
    options = {"node_size": 3000, "font_color": "white", "arrowsize": 20}
    positioning = nx.multipartite_layout(g, subset_key="layer")
    nx.draw_networkx(g, pos=positioning, with_labels=True, font_weight='bold', **options)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, positioning, edge_labels)  # shown later

    # browse the graph and build the mixed probability law
    mixed = build_mixed_law(g, start_node)
    definition = (10.0, 16.0)  # manually defined-
    mixed_phi = mixed.get_function()
    integral, error = scipy.integrate.quad(mixed_phi, *definition)
    mass_f = build_integral_function(lambda x: mixed_phi(x) / integral, definition)

    # plot density and mass functions
    fig, ax1 = mp.subplots()
    color = 'tab:green'
    ax1.set_xlabel("Duration [h]")
    ax1.set_ylabel("Density function", color=color)
    ax1.plot(*sample_function(mixed_phi, definition, samples_=1000), color=color)
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
    print(f"Variance: {variance:.2f}")
    mass_f_ninety = lambda x: mass_f(x) - 0.9
    ninety = scipy.optimize.fsolve(mass_f_ninety, [expected])  # start at the esperance value
    print(f"Maximum duration, with 90% probability: {ninety[0]:.2f} hours.")


if __name__ == '__main__': main()
