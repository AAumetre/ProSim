#!/usr/bin/env python3

import json
import os
import logging
import time
from collections import defaultdict

import matplotlib.pyplot as mp
from typing import *

import numpy
import scipy as scipy
from matplotlib.ticker import MultipleLocator

from GraphMonteCarlo import MonteCarlo
from Event import EventFactory
import networkx as nx

Function1D = Callable[[float], float]
DiscreteFunction1D = Tuple[List[float], List[float]]


def load_graph_json(path_: str):
    """ Load RiskAlternative.json and FlowDefinition.json and return all the necessary objets. """
    starting_node, ending_node, nodes, edges = None, None, {}, []
    # read tasks definitions
    with open(path_) as f:
        data = json.load(f)
        event_factory = EventFactory()
        for e in data["events"]:
            event = event_factory.create_event(e)
            nodes[event.name_] = event
        starting_node = nodes[data["starting_event"]]
        ending_node = nodes[data["ending_event"]]
        # read edges definition
        for e in data["weighted_edges"]:
            edges.append((nodes[e["from"]], nodes[e["to"]], e["p"]))
    # create the graph
    graph = nx.DiGraph()
    for name, node in nodes.items():
        graph.add_node(node, layer=node.layer_)
    graph.add_weighted_edges_from(edges)
    return starting_node, ending_node, graph


def print_graphs(sample_type_: str, sample_: List[float], graph_title_ : str = "") -> None:
    # plot density and mass functions
    fig, ax1 = mp.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(f"{sample_type_}, {len(sample_)} samples")
    ax1.set_ylabel("Density function", color=color)
    ax1.hist(sample_, bins=200, ls="dashed", alpha=0.8)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(which="both", visible=False)
    mp.yscale("log")
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Mass function", color=color)
    # compute the mass function
    bins = numpy.histogram(sample_, bins=len(sample_)//10)  # use numpy to binify the samples
    mass_function = scipy.integrate.cumulative_trapezoid(bins[0], bins[1][:-1])
    # normalize the mass function
    mass_function = [v/max(mass_function) for v in mass_function]
    mp.yscale("linear")
    # ax2.set_ylim([0, 1])  # in the end, not such a great idea
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.plot(bins[1][:-2], mass_function, "-", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    mp.title(graph_title_)


def print_simple_stats(name_: str, samples_: List[float], number_runs_: int):
    scale_factor = len(samples_)/number_runs_
    print(f"Basic statistics of {name_}, {len(samples_)} samples")
    print(f"\t* expectation for {name_}: {scale_factor*numpy.mean(samples_):.2f}.")
    print(f"\t* variance for {name_}: {numpy.var(samples_):.2f}.")
    print(f"\t* std deviation for {name_}: {numpy.std(samples_):.2f}.")
    quantiles = " - ".join([f"{q:.2f}" for q in [x*scale_factor for x in numpy.quantile(samples_, [0.25, 0.5, 0.75, 1.0])]])
    print(f"\t* quantiles for {name_}: {quantiles}")


def run_monte_carlo_analysis(json_file_path_: str, number_runs_: int, batch_mode_: bool = False) -> None:
    json_file_name = json_file_path_.split("/")[-1][:-5]  # split and remove the '.json' part
    mp.style.use("bmh")

    # Define a graph describing the process, starting with the tasks
    start_node, end_node, g = load_graph_json(json_file_path_)
    # find all the declared side effects data types
    declared_data_types = set()
    for n in g.nodes:
        for s in n.side_effects_:
            declared_data_types.add(s.type_)

    # do a Monte-Carlo simulation
    solver = MonteCarlo(g, start_node)
    mc_samples = solver.compute_samples(number_runs_)

    # create sublists for each sample type that can be found
    samples_sublists = defaultdict(list)
    for s in mc_samples:
        for s_type, s_value in s.items():
            samples_sublists[s_type].append(s_value)
    sample_types_str = ", ".join(samples_sublists.keys())
    print(f"Data sampled in this run: {sample_types_str}")
    # check that all the declared types have been sampled
    if declared_data_types != set(samples_sublists.keys()):
        logging.warning(f"Some side effect data types have been declared but are not captured"
                        f"in the samples: {declared_data_types.difference(set(samples_sublists.keys()))}")

    # draw the process graph
    options = {"node_size": 3000, "font_color": "black", "arrowsize": 20, "font_size": 12}
    # use the layer information to do the drawing
    positioning = nx.multipartite_layout(g, subset_key="layer")
    # positioning = nx.spring_layout(g, seed = 5)  # automatic positioning
    nx.draw_networkx(g, pos=positioning, with_labels=True, font_weight='bold', **options)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, positioning, edge_labels, label_pos=0.8)  # shown later

    # print all the samples' statistics graphs
    for sample_type, sample in samples_sublists.items():
        print_graphs(sample_type, sample)
        mp.savefig(json_file_name+"-"+sample_type+".png", dpi=180, format="PNG")
    if not batch_mode_: mp.show()  # shows both the graphs and the network diagram

    # print out some metadata
    print(f"{'='*100}\nStatistics for {json_file_name} ({number_runs_} runs)")
    for sample_name, sample in samples_sublists.items():
        print_simple_stats(sample_name, sample, number_runs_)


def main():
    # TODO: we might have to set requirements on input graphs topology (and automatically fix it later?)
    # TODO: compute transition probability sensitivity
    # TODO: improve logging to get a file log of all the results (+sample size per side-effect)
    # TODO: make the non-zero feature for normal distributions an option
    
    directory = "Examples"
    for filename in filter(lambda x: x.startswith("MultiOS"), os.listdir(directory)):
        run_monte_carlo_analysis(os.path.join(directory, filename), 100_000, True)
   

if __name__ == '__main__':
    start_time = time.time_ns()
    main()
    print(f"Duration: {(time.time_ns() - start_time) / 10 ** 9} s")
