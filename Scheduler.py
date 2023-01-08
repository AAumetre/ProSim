import queue

from Action import Action


class Scheduler:

    def __init__(self):
        self.time_: float = 0.0
        self.events_ = queue.PriorityQueue()  # time, order, Action
        self.kernel_ = None
        self.action_counter_: int = 0  # first scheduled, first executed
        # TODO: we could define an Action comparison method st. we can compare on priorities

    def __repr__(self):
        return f"{self.time_=}   {self.events_.qsize()} events in queue"

    def schedule(self, action_: Action, duration_: float):
        """ Enqueue work in the future. """
        self.events_.put((self.time_+duration_, self.action_counter_, action_))
        self.action_counter_ += 1

    def run_all(self):
        """ Finds the next action to run and runs it. Increments time."""
        while self.has_events():
            new_time, _, finished_action = self.events_.get_nowait()
            self.time_ = new_time
            # make Action's resources available again
            for res in finished_action.resources_:
                self.kernel_.available_resources_[res.type_] += res.qty_
            # make Action's outputs available
            for item in finished_action.outputs_:
                self.kernel_.items_in_stock_[item.type_] += item.qty_

    def has_events(self) -> bool:
        """ Tells whether there are events scheduled in the future. """
        return self.events_.qsize() > 0
