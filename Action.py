from typing import *
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

    def execute(self) -> Tuple[bool, Cost]:
        # at this point, the kernel made sure that:
        # all the inputs and all the resources are available
        for risk in self.events_.risks_:
            if risk.is_real():
                risk.recover()
                return False, risk.get_lost_cost()  # only one risk
        logging.debug(f"Action {self.name_} has been executed.")
        return True, self.events_.nominal_
