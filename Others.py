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
class RequestedResource:
    type_: str
    qty_: int | float


@dataclass()
class Resource(RequestedResource):
    properties_: Dict[str, str | int | float]
    anonymous_: bool = True

    def __add__(self, other):
        if self.type_ != other.type_ or not self.anonymous_:
            logging.error(f"Trying to add Resource {self} and {other}, which is not possible!")
            exit(1)
        return Resource(self.type_, self.qty_+other.qty_, self.properties_, True)

