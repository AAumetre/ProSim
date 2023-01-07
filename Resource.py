from Others import *


class Resource:

    def __init__(self, type_: str, qty_: int):
        self.type_ = type_
        self.qty_ = qty_
        self.busy_ = False


