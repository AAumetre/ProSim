from dataclasses import dataclass
from typing import *


@dataclass(frozen=True)
class Task:
    name_: str
    normal_: Tuple[float, float]
    layer_: int = 0

    def __repr__(self):
        return self.name_ + "\n" + str(self.normal_[0])
