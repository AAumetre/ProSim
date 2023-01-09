import collections
import math

from Action import Action
from Action import ActionImpact
from Others import *
from collections import defaultdict
from Scheduler import Scheduler
from Factory import Factory


class Kernel:

    def __init__(self, scheduler_: Scheduler):
        self.stochastic_mode: bool = False
        self.cookbook_: Dict[str, Dict[str,int|float]] = {}  # Item.type_, {name:Action.name_, qty:quantity}  # TODO: need a type here
        self.expected_items_: Dict[str, int] = defaultdict(int)  # Item.type_, count
        self.items_in_stock_: Dict[str, int] = defaultdict(int)
        self.available_resources_: Dict[str, List[Resource]] = {}
        self.actions_requirements_: Dict[str, ActionImpact] = {}  # Action.name_: ActionRequirements
        self.items_to_make_: List[ItemCount] = []
        self.actions_to_schedule_: List[str] = []  # contains Action names
        self.total_cost_ = Cost(0.0, 0.0)
        self.scheduler_ = scheduler_
        self.factory_: Factory = Factory()

    def __repr__(self):
        return f"KERNEL. Stock:{str(self.items_in_stock_)} Actions:{str(self.actions_to_schedule_)} " \
               f"Time: {self.total_cost_.duration_}"

    def run(self) -> bool:
        """ Runs for as long as possible. Returns true is there is nothing left to do. """
        can_run = True
        while can_run:
            could_schedule_action = self.try_scheduling_actions()
            at_least_one_change = self.try_producing_items()
            events_ran = self.scheduler_.run_all()
            can_run = could_schedule_action or at_least_one_change or events_ran

        # check whether everything is done
        if len(self.actions_to_schedule_) > 0 or len(self.items_to_make_) > 0:
            # TODO: report why, what was missing? Currently not really working
            logging.error(f"Impossible to make Items \"{self.items_to_make_}\"."
                          f"Actions left: \"{self.actions_to_schedule_}\"")
            return False
        return True

    def try_scheduling_actions(self) -> bool:
        """ Run all posted actions that can be done. """
        at_least_one_scheduled = False
        for action_name in self.actions_to_schedule_.copy():
            # check if all pre-requisites are met
            if self.action_can_execute(action_name)[0]:
                # create the Action
                action = self.factory_.create_action(action_name)
                # schedule it
                self.schedule_action(action)
                # remove the action from the queue
                self.actions_to_schedule_.remove(action_name)
                at_least_one_scheduled = True
        return at_least_one_scheduled

    def schedule_action(self, action_: Action):
        # take resources out of kernel and allocate them to Action
        for resource in action_.resources_:
            action_.allocate_resource(resource.type_, self.available_resources_[resource.type_].pop())

        # update Action's stochastic mode TODO: could be done at the creation of the Action
        action_.stochastic_mode_ = self.stochastic_mode
        # update cost
        success, cost_or_event = action_.compute_cost()
        if success:
            cost = cost_or_event
        else:
            cost = self.recover_from_risk(action_, cost_or_event)
        self.total_cost_ += cost

        # schedule end of action
        self.scheduler_.schedule(action_, cost.duration_)
        # update stocks
        for item in action_.inputs_:
            self.items_in_stock_[item.type_] -= item.qty_

    def try_producing_items(self) -> bool:
        """ At run-time, look for items that need to be produced. """
        if len(self.items_to_make_) == 0:
            return False

        for item in self.items_to_make_.copy():
            # if we do not have enough stock, schedule the required work
            if (self.items_in_stock_[item.type_] + self.expected_items_[item.type_]) < item.qty_:
                # find out how to make what's missing
                needed_action_name = self.get_recipe(item.type_)
                needed_action_impacts = self.actions_requirements_[needed_action_name]
                can_execute, missing_items = self.action_can_execute(needed_action_name)
                # how many of that Action should be scheduled?
                n_items_produced = self.cookbook_[item.type_]["qty"]
                n_actions_needed = math.ceil(item.qty_ / n_items_produced)

                # delegate the scheduling of the action, only enqueue it here
                for _ in range(n_actions_needed):
                    self.actions_to_schedule_.append(needed_action_name)
                # there are possibly missing items, try to make them
                for input_item in missing_items:
                    for _ in range(n_actions_needed):
                        self.produce_item(input_item)

                # decrease the number of needed items
                for _ in range(n_actions_needed):
                    self.reduce_items_needs(needed_action_impacts.outputs_)
                # increase the number of expected items
                for out in needed_action_impacts.outputs_:
                    out_items_produced = self.cookbook_[out.type_]["qty"]
                    self.expected_items_[out.type_] += n_actions_needed*out_items_produced
                # subtract expected of this item
                self.expected_items_[item.type_] -= item.qty_

            else:
                if self.items_in_stock_[item.type_] < item.qty_:  # we're counting on expected items
                    self.expected_items_[item.type_] -= item.qty_ - self.items_in_stock_[item.type_]
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
                while not found and abs(i_in_queue) <= len(self.items_to_make_):
                    if self.items_to_make_[i_in_queue].type_ == item.type_:
                        found = True
                        break
                    i_in_queue -= 1
                if not found:
                    # if the item is not found, it might be because it's an output of a scheduled Action
                    break
                if self.items_to_make_[i_in_queue].qty_ > items_produced:
                    self.items_to_make_[i_in_queue].qty_ -= items_produced
                    items_reduced = items_produced  # we're done
                else:
                    items_reduced += self.items_to_make_[i_in_queue].qty_
                    self.items_to_make_.remove(self.items_to_make_[i_in_queue])  # TODO: check which is deleted!

    def finish_action(self, action_: Action):
        # make Action's resources available again
        for resource_type, resource_list in action_.allocated_resources.items():
            self.available_resources_[resource_type] += action_.allocated_resources[resource_type]
        # make Action's outputs available
        if action_.success_:
            for item in action_.outputs_:
                self.items_in_stock_[item.type_] += item.qty_
                self.expected_items_[item.type_] = max(0, self.expected_items_[item.type_] - item.qty_)
        # TODO: check that after this, the Action is deleted

    def recover_from_risk(self, action_: Action, risk_: Risk) -> Cost:
        """ Defines what happens when an Action fails and a given Risk occurs. """
        # Items that should have been produced should be scheduled for production again
        self.actions_to_schedule_.append(action_)
        # Risk defines a list of other Actions to be scheduled
        for action_name in risk_.recovering_actions_:
            pass
            # TODO: replace self.actions_to_schedule_.append(self.known_actions_[action_name])
        # Risk also defines what is the lost Cost of the failure itself
        lost_duration = eval(risk_.lost_cost_.duration_, {"d": action_.events_.nominal_.duration_})
        lost_cost = eval(risk_.lost_cost_.cost_, {"c": action_.events_.nominal_.cost_})
        return Cost(lost_duration, lost_cost)

    def action_can_execute(self, action_name_: str) -> Tuple[bool, List[ItemCount]]:
        """ Checks whether an Action can be executed, if not, returns what is needed. """
        needed_items = []
        can_execute = True
        for item in self.actions_requirements_[action_name_].inputs_:
            if self.items_in_stock_[item.type_] >= item.qty_:  # TODO: unit conversion
                continue
            else:
                logging.debug(f"Item {item.type_} not in sufficient quantity.")
                needed_items.append(item)
                can_execute = False
        n_required_resources = defaultdict(int)
        for res in self.actions_requirements_[action_name_].resources_:  # TODO: should not be needed here
            # build a dict, containing req_type: count
            n_required_resources[res.type_] += 1  # TODO: should be done differently, method in ActionRequirements?
        for resource_type, count in n_required_resources.items():
            if not self.available_resources_[resource_type]:
                logging.debug(f"Resource {res.type_} is currently not available.")
                can_execute = False
                break
            if len(self.available_resources_[resource_type]) < count:
                logging.debug(f"Resource {res.type_} is currently not available.")
                can_execute = False
                break
        return can_execute, needed_items

    def get_recipe(self, item_name_: str) -> str:
        """ Looks for the Action that allows to produce a given item and returns its name. """
        if item_name_ in self.cookbook_:
            return self.cookbook_[item_name_]["name"]
        else:
            logging.error(f"Cannot produce Item \"{item_name_}\".")
            exit(1)

    def declare_action(self, action_: Action) -> None:
        """ Get information about an Action, its inputs, outputs and resources. """
        # update the cookbook
        for item in action_.outputs_:
            if item.type_ in self.cookbook_:
                # TODO: actually, handling alternate recipes is a feature that we might need
                logging.error(f"Kernel cannot add recipe \"{action_.name_}\" for \"{item.type_}\""
                              f"because a recipe already exists: {self.cookbook_[item.type_]}.")
                exit(1)
            else:
                self.cookbook_[item.type_] = {"name": action_.name_, "qty": action_.get_output_count(item.type_)}
        # update Action requirements dictionary
        self.actions_requirements_[action_.name_] = ActionImpact(action_)

    def produce_item(self, item_: ItemCount) -> None:
        """ Adds an item to produce right now to the queue. """
        if item_.type_ in self.cookbook_:
            self.items_to_make_.append(item_)
        else:
            logging.error(f"Requested item \"{item_.type_}\" is unknown.")

    def add_to_stock(self, item_: ItemCount) -> None:
        """ Adds an item to the stock. """
        self.items_in_stock_[item_.type_] += item_.qty_

    def add_resource(self, res_: Resource) -> None:
        """ Give a new resource to the Kernel. """
        if res_.type_ not in self.available_resources_:
            self.available_resources_[res_.type_] = []
        self.available_resources_[res_.type_].append(res_)

    def set_stochastic_mode(self, enabled_: bool) -> None:
        self.stochastic_mode = enabled_
