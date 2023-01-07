from typing import *
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
    action_: str
    probability: Callable
    lost_cost_: str

    def get_lost_cost(self, nominal_cost_: Cost) -> Cost:
        return Cost(0.0, 0.0)  # TODO: use defined cost function


@dataclass()
class Events:
    nominal_: Cost
    risks_: List[Risk]
