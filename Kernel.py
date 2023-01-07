import collections
import logging

from Action import Action
from Others import *
from collections import defaultdict
from Resource import Resource
from Scheduler import Scheduler


class Kernel:

    def __init__(self, scheduler_: Scheduler):
        self.actions_: Dict[str, Action] = {}
        self.items_in_stock_: Dict[str, int] = defaultdict(int)
        self.cookbook: Dict[str, Action] = {}  # Item.type_, Action
        self.items_queue_ = collections.deque()  # (Item, count)
        self.actions_queue_ = collections.deque()  # Action.name_ TODO get rid of this name?
        self.available_resources_: Dict[str, int] = defaultdict(int)
        self.existing_resources: Set[str] = set()
        self.total_cost = Cost(0.0, 0.0)
        self.scheduler_ = scheduler_

    def __repr__(self):
        return f"KERNEL. Stock:{str(self.items_in_stock_)} Actions:{str(self.items_queue_)} " \
               f"Time: {self.total_cost.duration_}"

    def run(self) -> bool:
        """ Runs for as long as possible. Returns true is there is nothing left to do. """
        can_run = True
        while can_run:
            at_least_one_change = self.try_scheduling_actions()
            at_least_one_change |= self.run_item()
            can_run = self.scheduler_.has_events() or at_least_one_change
            if not at_least_one_change:  # we cannot enqueue more event, run
                self.scheduler_.run_to_next()
        # check whether everything is done
        if len(self.actions_queue_) > 0 or len(self.items_queue_) > 0:
            return False
        return True

    def try_scheduling_actions(self) -> bool:
        """ Run all posted actions, until one cannot be done. """
        at_least_one_scheduled = False
        while len(self.actions_queue_) > 0:
            next_action = self.actions_[self.actions_queue_[-1]]
            # check if all pre-requisites are met
            if self.action_can_execute(next_action):
                self.schedule_action(next_action)
                # remove the action from the queue
                self.actions_queue_.pop()
                at_least_one_scheduled = True
            else:
                return at_least_one_scheduled
        return False

    def schedule_action(self, action_: Action):
        # update cost
        cost = action_.compute_cost()
        self.total_cost += cost
        # schedule end of action
        self.scheduler_.schedule(action_, cost.duration_)
        # make resources unavailable
        for res in action_.resources_:
            self.available_resources_[res.type_] -= res.qty_
        # update stocks
        for item in action_.inputs_:
            self.items_in_stock_[item.type_] -= item.qty_

    def run_item(self) -> bool:
        """ At run-time, look for items that need to be produced. """
        if len(self.items_queue_) == 0:
            return False
        item_needed, count = self.items_queue_[-1]
        if self.items_in_stock_[item_needed] < count:
            # find out how to make what's missing
            # TODO: we could add all the needed items, not do it one-by-one and enqueue enough actions to pass
            needed_action = self.get_recipe(item_needed)
            if self.action_can_execute(needed_action):
                self.schedule_action(needed_action)
                # decrease the number of needed items
                items_produced = 0
                # TODO: to be improved, change outputs to being a dict?
                for it in needed_action.outputs_:
                    if it.type_ == item_needed:
                        items_produced = it.qty_
                self.items_queue_[-1] = (item_needed, count-items_produced)
                return True
            else:
                # Action was needed but could not be scheduled
                self.actions_queue_.append(needed_action.name_)
                return False
        else:
            self.items_queue_.pop()
            return True

    def get_recipe(self, item_name_: str) -> Action:
        """ Looks for the action that allows to produce a given item. """
        if item_name_ in self.cookbook:
            return self.cookbook[item_name_]
        else:
            logging.error(f"Cannot produce Item \"{item_name_}\".")
            exit(1)

    def add_action(self, action_: Action) -> None:
        """ Add a possible Action to the dictionary. """
        if action_.name_ in self.actions_:
            logging.error(f"Kernel cannot add Action \"{action_.name_}\" because an Action with the same"
                          f"name already exists: {self.actions_[action_.name_]}.")
            exit(1)
        else:
            # register the new action
            self.actions_[action_.name_] = action_
            # update the cookbook
            for item in action_.outputs_:
                if item.type_ in self.cookbook:
                    logging.error(f"Kernel cannot add recipe \"{action_.name_}\" for \"{item.type_}\""
                                  f"because a recipe already exists: {self.cookbook[item.type_]}.")
                    exit(1)
                else:
                    self.cookbook[item.type_] = action_
            # update known resources
            for res in action_.resources_:
                self.existing_resources.add(res.type_)

    def action_can_execute(self, action_: Action) -> bool:
        """ Checks whether an Action can be executed, if not, queues what is needed. """
        for item in action_.inputs_:
            if self.items_in_stock_[item.type_] >= item.qty_:  # TODO: unit conversion
                logging.debug(f"Item {item.type_} in sufficient quantity, no need to produce it.")
                continue
            else:
                logging.debug(f"Item {item.type_} not in sufficient quantity.")
                self.produce_item(item)
                return False
        for res in action_.resources_:
            if self.available_resources_[res.type_] < res.qty_:
                logging.debug(f"Resource {res.type_} is currently not available.")
                return False
        return True

    def produce_item(self, item_: ItemCount) -> None:
        """ Adds an item to produce right now to the queue. """
        self.items_queue_.append((item_.type_, item_.qty_))

    def add_item_for_later(self, item_type_: str, count_: int) -> None:
        """ Adds an item to be produced after everything else is done. """
        if item_type_ in self.cookbook:
            self.items_queue_.appendleft((item_type_, count_))
        else:
            logging.error(f"Requested item \"{item_type_}\" is unknown.")

    def add_to_stock(self, item_: ItemCount) -> None:
        """ Adds an item to the stock. """
        self.items_in_stock_[item_.type_] += item_.qty_

    def add_resource(self, res_: Resource) -> None:
        """ Give a new resource to the Kernel. """
        self.available_resources_[res_.type_] += res_.qty_
        self.existing_resources.add(res_.type_)
