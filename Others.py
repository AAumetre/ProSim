from typing import *
import logging

from dataclasses import dataclass


@dataclass()
class ItemCount:
    type_: str
    qty_: int | float
    unit_: str


@dataclass()
class Cost:
    duration_: float
    cost_: float

    def __add__(self, other):
        return Cost(self.duration_+other.duration_, self.cost_+other.cost_)


@dataclass()
class Risk:
    name_: str
    recovering_actions_: List[str]  # actions names to be called to recover from the failure
    probability: str  # probability function as evaluable expression
    lost_cost_: Cost


@dataclass()
class Events:
    nominal_: Cost
    risks_: List[Risk]


@dataclass()
class Resource:
    type_: str
    qty_: int
