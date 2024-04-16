from collections import defaultdict
from typing import *
from numpy import random
from networkx import DiGraph

NodeType = Any


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
            samples.append(self.run_round())
        return samples

    def run_round(self) -> Dict[str, float]:
        effects = defaultdict(float)
        current_node = self.start_node_
        while True:
            successors = self.graph_.succ[current_node]
            if not successors:
                break
            current_node = self.choose_successor(successors)
            for effect_type, effect_value in current_node.trigger_event().items():
                effects[effect_type] += effect_value
        return effects

    def converge_samples(self, threshold_: float = 0.1) -> Tuple[Dict[str, float], int]:
        """ Computes samples and tries converging on one of the data types. """
        pass

    def choose_successor(self, successors_: Dict) -> NodeType:
        """ Choose a successor, based on a uniform density in [0.0 ; 1.0]. """
        names, probabilities = [], []
        for successor, keys in successors_.items():
            names.append(successor)
            probabilities.append(keys["weight"])
        return random.choice(names, p=probabilities)