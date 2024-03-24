from collections import defaultdict
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


class MonteCarlo:

    def __init__(self, graph_: DiGraph, start_node_: NodeType, seed_: int = 666):
        self.graph_ = graph_
        self.start_node_ = start_node_
        self.seed_ = seed_
        self.gen_ = random.default_rng(seed_)

    def compute_samples(self, sample_size_: int) -> Dict[str, float]:
        samples = []
        # run through the graph
        for i in range(sample_size_):
            samples.append(defaultdict(float))
            current_node = self.start_node_
            while True:
                successors = self.graph_.succ[current_node]
                if not successors:
                    break
                current_node = choose_successor(successors)
                for effect_type, effect_value in current_node.trigger_event().items():
                    samples[i][effect_type] += effect_value
        return samples
