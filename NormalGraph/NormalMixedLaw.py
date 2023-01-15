import math
from dataclasses import dataclass
from typing import *

Function1D = Callable[[float], float]


@dataclass()
class NormalMixedLaw:
    weighted_normals_: Dict[float, Tuple[float, float]]  # p, m, s
    normals_: Tuple[float, float]

    def __add_normal(self, n1_: Tuple[float, float], n2_: Tuple[float, float]) -> Tuple[float, float]:
        s = math.sqrt(n1_[1] ** 2 + n2_[1] ** 2)
        return n1_[0] + n2_[0], s

    def add_normal(self, dist_: Tuple[float, float, float]):
        p, m, s = dist_
        if p == 1.0:
            self.normals_ = self.__add_normal(self.normals_, (m, s))
        else:
            if p in self.weighted_normals_:
                self.weighted_normals_[p] = self.__add_normal(self.weighted_normals_[p], (m, s))
            else:
                self.weighted_normals_[p] = (m, s)

    def get_function(self) -> Function1D:
        def f(x_: float) -> float:
            y = normal_dist(x_, *self.normals_)
            for p, dist in self.weighted_normals_.items():
                m, s = dist
                y += p * normal_dist(x_, m+self.normals_[0], s)
            return y
        return f


def normal_dist(x_: float, m_: float, s_: float) -> float:
    return (1 / (s_ * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x_ - m_) / s_) ** 2)
