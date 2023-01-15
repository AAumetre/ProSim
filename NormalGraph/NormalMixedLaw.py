import math
from typing import *
import scipy

Function1D = Callable[[float], float]
NormalDensity = Tuple[float, float, float]  # p, m, s


class NormalMixedLaw:

    def __init__(self, phi_dx_: float):
        self.phi_: List[float] = [1.0]  # discrete dirac
        self.phi_dx_: float = phi_dx_
        self.phi_def_: Tuple[float, float] = (0.0, phi_dx_)
        self.epsilon_: float = 1e-6
        self.samples_left_: int = 0

    def add_normal_densities(self, distributions_: List[NormalDensity]):
        if not distributions_: return
        # build the density function, either a pure normal or a composed one
        new_def = self.phi_def_
        for dist in distributions_:
            p, m, s = dist
            bounds = (m - 5 * s, m + 5 * s)  # 5-sigmas on each side
            new_def = (min(new_def[0], bounds[0]), max(new_def[1], bounds[1]))
        new_density = self.build_distribution(distributions_)
        # sample the new density function
        _, new_samples = sample_function(new_density, new_def, self.phi_dx_)
        # zero-pad phi to match the new size
        zeros_left = [0.0] * int(abs(self.phi_def_[0] - new_def[0]) / self.phi_dx_)
        zeros_right = [0.0] * int(abs(self.phi_def_[1] - new_def[1]) / self.phi_dx_)
        # update phi with a discrete convolution product
        self.phi_ = list(scipy.signal.convolve(zeros_left + self.phi_ + zeros_right, new_samples) / sum(self.phi_))
        self.phi_def_ = new_def
        self.clean_up_phi()

    def clean_up_phi(self):
        """ Remove parts of phi which are below epsilon. """
        i = 0
        while i < len(self.phi_):
            if self.phi_[i] > self.epsilon_:
                self.samples_left_ += i
                self.phi_ = self.phi_[i:]
                self.phi_def_ = (0.0, self.phi_def_[1] - i*self.phi_dx_)
                break
            i += 10
        i = len(self.phi_) - 1
        while i > 0:
            if self.phi_[i] > self.epsilon_:
                self.phi_ = self.phi_[:i]
                self.phi_def_ = (0.0, len(self.phi_)*self.phi_dx_)
                break
            i -= 10

    def build_distribution(self, distributions_: List[NormalDensity]) -> Function1D:
        def phi(x_: float) -> float:
            y = 0.0
            for dist in distributions_:
                p, m, s = dist
                y += p * normal_dist(x_, m, s)
            return y
        return phi

    def get_sampling(self) -> Tuple[List[float], List[float]]:
        first_non_zero = self.phi_def_[0] + self.samples_left_*self.phi_dx_
        x = [first_non_zero+i*self.phi_dx_ for i in range(len(self.phi_))]
        return x, self.phi_


def normal_dist(x_: float, m_: float, s_: float) -> float:
    return (1 / (s_ * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x_ - m_) / s_) ** 2)


def sample_function(f_: Function1D, def_: Tuple[float, float], dx_: float = 0.01) -> Tuple[List[float], List[float]]:
    """ Compute the [y], y=f_(x) arrays. """
    min_x, max_x = def_[0], def_[1]
    n_samples = int((max_x - min_x) / dx_)
    xs, ys = [0.0] * n_samples, [0.0] * n_samples
    for i in range(n_samples):
        xs[i] = min_x + i * dx_
        ys[i] = f_(xs[i])
    return xs, ys
