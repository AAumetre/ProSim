from typing import *
from numpy import random
from networkx import DiGraph

NodeType = Any


def choose_successor(successors_: Dict) -> NodeType:
    """ Choose a successor, based on a uniform density in [0.0 ; 1.0]. """
    names, probabilities = [], []
    for successor, keys in successors_.items():
        names.append(successor)
        probabilities.append(keys["weight"])
    return random.choice(names, p=probabilities)


def get_node_value(node_: NodeType, random_on_nodes_: bool):
    if not random_on_nodes_:
        return node_.normal_[0]  # mean value
    else:
        return random.normal(*node_.normal_)


class MonteCarlo:

    def __init__(self, graph_: DiGraph, start_node_: NodeType, seed_: int = 666):
        self.graph_ = graph_
        self.start_node_ = start_node_
        self.seed_ = seed_
        self.gen_ = random.default_rng(seed_)

    def compute_sample(self, size_: int, random_on_nodes_: bool = True) -> List[float]:
        samples = [0.0] * size_
        for i in range(size_):
            samples[i] = self.run_round(random_on_nodes_)
        return samples

    def run_round(self, random_on_nodes_: bool = True) -> float:
        """ Run through the graph once, taking into account random variables or not. """
        current_node = self.start_node_
        total = get_node_value(current_node, random_on_nodes_)
        while True:
            successors = self.graph_.succ[current_node]
            if not successors:
                break
            current_node = choose_successor(successors)
            total += get_node_value(current_node, random_on_nodes_)
        return total
