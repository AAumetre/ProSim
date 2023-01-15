import glob
import json
from collections import defaultdict

from Action import Action
from Others import *
from typing import *


class Factory:

    def create_action(self, name_: str) -> Action:
        """ Tries creating an Action of type name_ and returns it. """
        raise NotImplementedError()

    def create_resource(self, type_) -> Resource:
        """ Tries creating a Resource of type type_ and returns it. """
        raise NotImplementedError()


class JsonFactory(Factory):

    def __init__(self):
        self.action_definition_path: Dict[str, str] = {}  # Action.name_, path_to_json_definition
        self.resource_definition_path: Dict[str, List[str]] = defaultdict(list)  # Resource.type_, path_to_json_definition

    def create_action(self, name_: str) -> Action:
        return self.__create_action_from_json(self.action_definition_path[name_])

    def create_resource(self, type_) -> Resource:
        # TODO: should not be used, to be re-considered
        return self.create_resource_from_json(self.resource_definition_path[type_][0])

    def inspect_folder_for_actions(self, path_: str):
        """ Looks into a folder for JSON files defining Actions and records them. """
        action_paths = glob.glob(path_ + "/*.json")
        for ap in action_paths:
            action = self.__create_action_from_json(ap)
            self.action_definition_path[action.name_] = ap

    def inspect_folder_for_resources(self, path_: str):
        """ Looks into a folder for JSON files defining Resources and records them. """
        res_paths = glob.glob(path_ + "/*.json")
        for rp in res_paths:
            resource = self.create_resource_from_json(rp)
            self.resource_definition_path[resource.type_].append(rp)

    def create_resource_from_json(self, path_: str) -> Resource:
        with open(path_, "r") as f:
            json_data = json.load(f)
            return self.__read_resource(json_data)

    def __create_action_from_json(self, path_: str):
        """ Create an Action from a JSON file. """
        with open(path_, "r") as f:
            json_data = json.load(f)
            return self.__create_action(json_data)

    def __create_action(self, d_: Dict) -> Action:
        inputs, outputs, resources = [], [], []
        for e in d_["inputs"]:
            inputs.append(self.__read_itemcount(e))
        for e in d_["outputs"]:
            outputs.append(self.__read_itemcount(e))
        for e in d_["resources"]:
            resources.append(self.__read_resource(e))
        events = self.__read_events(d_["events"])
        duration_v = d_["events"]["duration_variability"]
        cost_v = d_["events"]["cost_variability"]
        return Action(d_["name"], inputs, outputs, resources, events, duration_v, cost_v)

    def __read_itemcount(self, d_: Dict) -> ItemCount:
        return ItemCount(d_["type"], d_["qty"], d_["unit"])

    def __read_events(self, d_: Dict) -> Events:
        nominal = Cost(d_["nominal"]["duration"], d_["nominal"]["cost"])
        risks = []
        for r in d_["risks"]:
            risks.append(Risk(r["name"], r["actions"], r["probability"],
                              Cost(r["lost_cost"]["duration"], r["lost_cost"]["cost"])))
        return Events(nominal, risks)

    def __read_resource(self, d_: Dict) -> Resource:
        if "properties" in d_:
            return Resource(d_["type"], d_["properties"])
        else:
            return Resource(d_["type"])
