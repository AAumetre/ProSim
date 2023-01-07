import collections
import logging
from Action import Action
from Others import *
from collections import defaultdict
from Resource import Resource


class Kernel:

    def __init__(self):
        self.actions_: Dict[str, Action] = {}
        self.items_in_stock_: Dict[str, int] = defaultdict(int)
        self.cookbook: Dict[str, Action] = {}  # Item.type_, Action
        self.items_queue_ = collections.deque()  # (Item, count)
        self.actions_queue_ = collections.deque()  # Action.name_
        self.available_resources_: Dict[str, int] = defaultdict(int)
        self.existing_resources: Set[str] = set()
        self.total_cost = Cost(0.0, 0.0)

    def __repr__(self):
        return f"KERNEL. Stock:{str(self.items_in_stock_)} Actions:{str(self.items_queue_)} Time: {self.total_cost.duration_}"

    def run(self) -> bool:
        """ Runs for as long as possible. Returns true is there is nothing left to do. """
        can_run = True
        while (len(self.actions_queue_) > 0 or len(self.items_queue_) > 0) and can_run:
            could_run_action = self.run_actions()
            self.run_item()
            can_run = could_run_action or (self.actions_queue_[-2] != self.actions_queue_[-1])  # TODO: fishy
        if len(self.actions_queue_) > 0 or len(self.items_queue_) > 0:
            return False
        return True

    def run_actions(self) -> bool:
        """ Run all posted actions, until one cannot be done. """
        while len(self.actions_queue_) > 0:
            next_action = self.actions_[self.actions_queue_[-1]]
            # check if all pre-requisites are met
            if self.action_can_execute(next_action):
                _, cost = next_action.execute()
                self.total_cost += cost
                self.actions_queue_.pop()
                # update stocks
                for item in next_action.inputs_:
                    self.items_in_stock_[item.type_] -= item.qty_
                for item in next_action.outputs_:
                    self.items_in_stock_[item.type_] += item.qty_
            else:
                return False
        return True

    def run_item(self) -> bool:
        """ At run-time, look for items that need to be produced. """
        item_needed, count = self.items_queue_[-1]
        if self.items_in_stock_[item_needed] < count:
            # find out how to make what's missing
            self.actions_queue_.append(self.get_recipe(item_needed).name_)
            # TODO: we could add all the needed items, not do it one-by-one and enqueue enough actions to pass
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
