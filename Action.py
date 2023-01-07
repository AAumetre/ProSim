from Others import *
from Resource import Resource


class Action:

    def __init__(self, name_: str, inputs_: List[ItemCount], outputs: List[ItemCount],
                 resources_: List[Resource], events_: Events):
        self.name_ = name_
        self.inputs_ = inputs_
        self.outputs_ = outputs
        self.resources_ = resources_
        self.events_ = events_
        self.kernel_ = None

    def __repr__(self):
        return self.name_

    def compute_cost(self) -> Cost:
        # at this point, the kernel made sure that:
        # all the inputs and all the resources are available
        for risk in self.events_.risks_:
            pass
        logging.debug(f"Action {self.name_} has been executed.")
        return self.events_.nominal_
