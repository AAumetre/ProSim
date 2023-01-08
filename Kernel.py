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
        self.cookbook: Dict[str, Action] = {}  # Item.type_, Action
        self.items_in_stock_: Dict[str, int] = defaultdict(int)
        self.available_resources_: Dict[str, int] = defaultdict(int)
        self.items_queue_ = collections.deque()  # ItemCount
        self.actions_queue_ = collections.deque()  # Action TODO get rid of this name?
        self.existing_resources: Set[str] = set()
        self.total_cost = Cost(0.0, 0.0)
        self.scheduler_ = scheduler_

    def __repr__(self):
        return f"KERNEL. Stock:{str(self.items_in_stock_)} Actions:{str(self.actions_queue_)} " \
               f"Time: {self.total_cost.duration_}"

    def run(self) -> bool:
        """ Runs for as long as possible. Returns true is there is nothing left to do. """
        can_run = True
        while can_run:
            at_least_one_change = True
            while at_least_one_change:
                at_least_one_change = self.try_scheduling_actions()
                at_least_one_change |= self.try_producing_items()
            can_run = self.scheduler_.has_events() or at_least_one_change
            self.scheduler_.run_to_next()
        # check whether everything is done
        if len(self.actions_queue_) > 0 or len(self.items_queue_) > 0:
            # TODO: report why, what was missing? Currently not really working
            logging.error(f"Impossible to make Items \"{self.items_queue_}\"."
                          f"Actions left: \"{self.actions_queue_}\"")
            return False
        return True

    def try_scheduling_actions(self) -> bool:
        """ Run all posted actions that can be done. """
        at_least_one_scheduled = False
        for action in self.actions_queue_.copy():
            # check if all pre-requisites are met
            if self.action_can_execute(action)[0]:
                self.schedule_action(action)
                # remove the action from the queue
                self.actions_queue_.remove(action)
                at_least_one_scheduled = True
        return at_least_one_scheduled

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

    def try_producing_items(self) -> bool:
        """ At run-time, look for items that need to be produced. """
        if len(self.items_queue_) == 0:
            return False
        item_needed: ItemCount = self.items_queue_.pop()
        self.items_queue_.append(item_needed)
        # TODO: this way of popping and possibly pushing with less elements effectively launches the tasks
        #       all in parallel, whereas we might want a "start as soon as possible" way
        # if we do not have enough, schedule the work
        if self.items_in_stock_[item_needed.type_] < item_needed.qty_:
            # find out how to make what's missing
            needed_action = self.get_recipe(item_needed.type_)
            can_execute, missing_items = self.action_can_execute(needed_action)
            # delegate the scheduling of the action, only enqueue it here
            self.actions_queue_.append(needed_action)
            # there are possibly missing items, try to make them
            for item in missing_items:
                self.items_queue_.append(item)
            # decrease the number of needed items
            self.reduce_items_needs(needed_action.outputs_)

        return True

    def reduce_items_needs(self, enqueued_: List[ItemCount]) -> None:
        """ Given a list of items that should be produced, reduce or pop elements from the item queue. """
        for item in enqueued_:
            items_produced = item.qty_
            items_reduced = 0
            # reduce starting from the end of the queue, there are possibly several elements to update
            while items_reduced < items_produced:
                i_in_queue = -1
                found = False
                # look for the most recent item record in the queue
                while not found and abs(i_in_queue) <= len(self.items_queue_):
                    if self.items_queue_[i_in_queue].type_ == item.type_:
                        found = True
                        break
                    i_in_queue -= 1
                if not found:
                    # if the item is not found, it might be because it's an output of a scheduled Action
                    break
                if self.items_queue_[i_in_queue].qty_ > items_produced:
                    self.items_queue_[i_in_queue].qty_ -= items_produced
                    items_reduced = items_produced  # we're done
                else:
                    items_reduced += self.items_queue_[i_in_queue].qty_
                    self.items_queue_.remove(self.items_queue_[i_in_queue])  # TODO: check which is deleted!

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

    def action_can_execute(self, action_: Action) -> Tuple[bool, List[ItemCount]]:
        """ Checks whether an Action can be executed, if not, queues what is needed. """
        needed_items = []
        can_execute = True
        for item in action_.inputs_:
            if self.items_in_stock_[item.type_] >= item.qty_:  # TODO: unit conversion
                continue
            else:
                logging.debug(f"Item {item.type_} not in sufficient quantity.")
                needed_items.append(item)
                can_execute = False
        for res in action_.resources_:
            if self.available_resources_[res.type_] < res.qty_:
                logging.debug(f"Resource {res.type_} is currently not available.")
                can_execute = False
        return can_execute, needed_items

    def produce_item(self, item_: ItemCount) -> None:
        """ Adds an item to produce right now to the queue. """
        self.items_queue_.append(item_.type_)

    def add_item_for_later(self, item_: ItemCount) -> None:
        """ Adds an item to be produced after everything else is done. """
        if item_.type_ in self.cookbook:
            self.items_queue_.appendleft(item_)
        else:
            logging.error(f"Requested item \"{item_.type_}\" is unknown.")

    def add_to_stock(self, item_: ItemCount) -> None:
        """ Adds an item to the stock. """
        self.items_in_stock_[item_.type_] += item_.qty_

    def add_resource(self, res_: Resource) -> None:
        """ Give a new resource to the Kernel. """
        self.available_resources_[res_.type_] += res_.qty_
        self.existing_resources.add(res_.type_)
