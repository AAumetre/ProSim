from Others import *
from numpy import random


class Action:

    def __init__(self, name_: str, inputs_: List[ItemCount], outputs: List[ItemCount],
                 resources_: List[Resource], events_: Events):
        self.name_: str = name_
        self.inputs_: List[ItemCount] = inputs_
        self.outputs_: List[ItemCount] = outputs
        self.resources_: List[Resource] = resources_
        self.events_ = events_
        self.stochastic_mode_: bool = False
        self.success_: bool = True

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
                    return True, self.events_.nominal_
                else:
                    self.success_ = False
                    return False, self.events_.risks_[index]

    def get_output_count(self, type_: str) -> int:
        """ Finds the matching Item in its output, if any, and returns how many it produces. """
        for out in self.outputs_:
            if out.type_ == type_:
                return out.qty_
        return 0
