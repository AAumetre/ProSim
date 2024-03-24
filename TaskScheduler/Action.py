from Others import *
from numpy import random


class Action:

    def __init__(self, inputs_: List[ItemCount], outputs: List[ItemCount],
                 resources_: List[Resource], events_: Events,
                 duration_variability_: str, cost_variability_: str):
        self.name_: str = name_
        self.inputs_: List[ItemCount] = inputs_
        self.outputs_: List[ItemCount] = outputs
        self.resources_: List[Resource] = resources_
        self.allocated_resources: Dict[str, List[Resource]] = {}
        self.events_ = events_
        self.stochastic_mode_: bool = False
        self.success_: bool = True
        self.duration_variability_: str = duration_variability_
        self.cost_variability_: str = cost_variability_

    def __repr__(self):
        return self.name_

    def compute_cost(self) -> Tuple[bool, Cost | Risk]:
        """ Selects what happens and returns the corresponding status and cost. """
        if not self.stochastic_mode_:
            self.success_ = True
            return True, self.events_.nominal_

        # compute the probability of all the risks
        nominal_event_probability = 1.0
        events_ranges = [0.0]
        for risk in self.events_.risks_:
            p = eval(risk.probability)
            events_ranges.append(events_ranges[-1] + p)
            nominal_event_probability -= p
        events_ranges.append(1.0)  # upper-bound for nominal event
        if nominal_event_probability <= 0.0:
            logging.error(f"Nominal event probability of Action \"{self.name_}\" has gone below zero.")
            exit(1)

        # roll the dices and compute the event that occurs
        dice_value = random.random()
        for index, upper_bound in enumerate(events_ranges):
            if dice_value <= upper_bound:
                if index >= len(self.events_.risks_):  # this is the nominal case
                    self.success_ = True
                    if "Cook" in self.allocated_resources.keys():  # TODO: not really the intended use
                        # actual_duration = eval(self.duration_variability_, {"experience": self.properties_["experience"]})
                        cook_experience = self.allocated_resources["Cook"][0].properties_["experience"]
                        actual_duration = self.events_.nominal_.duration_*random.normal(eval(self.duration_variability_,
                                                             {"experience": cook_experience}), 1.0)
                        return True, Cost(actual_duration, self.events_.nominal_.cost_)
                    return True, self.events_.nominal_
                else:
                    self.success_ = False
                    return False, self.events_.risks_[index-1]

    def get_output_count(self, type_: str) -> int:
        """ Finds the matching Item in its output, if any, and returns how many it produces. """
        for out in self.outputs_:
            if out.type_ == type_:
                return out.qty_
        return 0

    def allocate_resource(self, type_: str, resource_: Resource):
        if type_ not in self.allocated_resources:
            self.allocated_resources[type_] = []
        self.allocated_resources[type_].append(resource_)


class ActionImpact:

    def __init__(self, action_: Action):
        # TODO: handle units
        self.inputs_: List[Dict[str:str|int|float]] = action_.inputs_.copy()
        self.outputs_: List[Dict[str:str|int|float]] = action_.outputs_.copy()
        self.resources_: List[Dict[str:str]] = action_.resources_.copy()

