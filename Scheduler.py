import queue

from Action import Action


class Scheduler:

    def __init__(self):
        self.time_: float = 0.0
        self.events_ = queue.PriorityQueue()  # time, Action
        self.kernel_ = None

    def __repr__(self):
        return f"{self.time_=}   {self.events_.qsize()} events in queue"

    def schedule(self, action_: Action, duration_: float):
        """ Enqueue work in the future. """
        self.events_.put((self.time_+duration_, action_))

    def run_to_next(self):
        """ Finds the next action to run and runs it. Increments time."""
        if self.has_events():
            new_time, finished_action = self.events_.get_nowait()
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
