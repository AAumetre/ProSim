import logging
import math
from abc import abstractmethod
from collections import defaultdict
from typing import *

import numpy.random

class SideEffect:
    @abstractmethod
    def __init__(self, type_: str):
        self.type_: str = type_

    @abstractmethod
    def get_value(self) -> float:
        pass

class FixedSideEffect(SideEffect):
    """ a FixedSideEffect value is fixed: it will always be the same outcome. """
    def __init__(self, type_: str, value_: float):
        super().__init__(type_)
        self.value_ = value_

    def __repr__(self):
        return f"{self.type_}, fixed: {self.value_}"

    def get_value(self) -> float:
        return self.value_


class NormalSideEffect(SideEffect):
    """ a NormalSideEffect value is determined by sampling a normal distribution. """
    def __init__(self, type_: str, mean_: float, sd_: float):
        super().__init__(type_)
        self.mean_ = mean_
        self.sd_ = sd_

    def __repr__(self):
        return f"{self.type_}, normal: [{self.mean_}, {self.sd_}]"

    def get_value(self) -> float:
        val = numpy.random.normal(self.mean_, self.sd_)
        tries, max_tries = 0, 1000
        while val < 0.0:
            val = numpy.random.normal(self.mean_, self.sd_)
            tries += 1
            if tries > max_tries:
                logging.warning(f"Normal side effect ({self.mean_}, {self.sd_}) could not produce a positive value after {max_tries} tries. Using the mean value.")
                return self.mean_
        return val

class ThreePointsSideEffect(SideEffect):
    """ the ThreePointsSideEffect is using a triangular distribution, based on min, mean and max estimated values. """

    def __init__(self, type_: str, three_points_: List[float]):
        super().__init__(type_)
        self.three_points_ = sorted(list(three_points_))

    def __repr__(self):
        return f"{self.type_}, three-points: {self.three_points_}"
        
    def get_value(self) -> float:
        return numpy.random.triangular(*self.three_points_)

class Event:
    """ An Event is defined by a list of side effects, typically costs and durations. """
    def __init__(self, name_: str, layer_: int, side_effects_: List[SideEffect]):
        self.name_ = name_
        self.layer_ = layer_
        self.side_effects_ = side_effects_

    def __repr__(self):
        side_effects_str = ",\n".join(map(str, self.side_effects_))
        return f"{self.name_}\n{side_effects_str}"

    def trigger_event(self) -> Dict[str, float]:
        effect = defaultdict(float)
        for side_effect in self.side_effects_:
            effect[side_effect.type_] += side_effect.get_value()
        return effect

class EventFactory:
    """ Simple factory to read a JSON entry and create the corresponding event and side effects, if any. """
    def __init__(self):
        self.supported_side_effects_ = {"fixed", "normal", "three-points"}

    def create_event(self, json_def_: Dict) -> Event:
        event_name = json_def_["name"]
        event_layer = json_def_["layer"]
        side_effects = []
        for se_type, se_args in json_def_["side_effects"].items():
            for se_class, se_params in se_args.items():
                if se_class == "fixed":
                    side_effects.append(FixedSideEffect(se_type, float(se_params)))
                elif se_class == "normal":
                    side_effects.append(NormalSideEffect(se_type, se_params[0], se_params[1]))
                elif se_class == "three-points":
                    side_effects.append(ThreePointsSideEffect(se_type, se_params))
                else:
                    logging.error(f"Impossible to create event '{event_name}' based on provided record: [{se_type}]: {se_args}")
                    exit(1)
        return Event(event_name, event_layer, side_effects)
