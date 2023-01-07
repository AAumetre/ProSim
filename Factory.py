import json

from Action import Action
from Others import *
from typing import *
from Resource import Resource


class Factory:

    def create_action_from_json(self, path_: str):
        """ Create an Action from a JSON file. """
        with open(path_, "r") as f:
            json_data = json.load(f)
            return self.create_action(json_data)

    def create_action(self, d_: Dict) -> Action:
        inputs, outputs, resources = [], [], []
        for e in d_["inputs"]:
            inputs.append(self.read_itemcount(e))
        for e in d_["outputs"]:
            outputs.append(self.read_itemcount(e))
        for e in d_["resources"]:
            resources.append(self.read_resource(e))
        events = self.read_events(d_["events"])
        return Action(d_["name"], inputs, outputs, resources, events)

    def read_itemcount(self, d_: Dict) -> ItemCount:
        return ItemCount(d_["type"], d_["qty"], d_["unit"])

    def read_resource(self, d_: Dict) -> Resource:
        return Resource(d_["type"], d_["qty"])

    def read_events(self, d_: Dict) -> Events:
        return Events(Cost(d_["nominal"]["duration"], d_["nominal"]["cost"]), [])
