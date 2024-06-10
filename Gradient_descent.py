from typing import Callable

from Function import Function
from math import *
import numpy as np
from scipy.optimize import approx_fprime


class AdamParams:
    def __init__(self, learning_rate: float = 1e-4,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.time = 0


def update_adam(params: AdamParams, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if params.m is None:
        params.m = np.zeros_like(grad)
    if params.v is None:
        params.v = np.zeros_like(grad)

    params.time += 1
    params.m = params.beta1 * params.m + (1 - params.beta1) * grad
    params.v = params.beta2 * params.v + (1 - params.beta2) * grad ** 2

    m = params.m / (1 - params.beta1 ** params.time)
    v = params.v / (1 - params.beta2 ** params.time)

    adjusted_grad = m / (np.sqrt(v) + params.epsilon)
    return x - params.learning_rate * adjusted_grad


def main():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 - z * (y + x**2)")
    constraint = Function("y + x**2")

    """
    Here we can see, that first 4 starting points end up with arguably good minimum value, but 5th is going bad :(
    """
    starting_points = [
        [0.103, -0.01, 0],
        [0.05, -0.05, 0],
        [-0.01, 0.01, 0]
        # [1, -1, 1.2],
        # [0.23, 0.32, 0]
    ]

    for start in starting_points:
        result = gradient_descend(rosenbrok_lagrange, constraint, start, 1e-3)
        ans = result[0]
        value = rosenbrok_lagrange.compute(ans)
        print()
        print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")
        print(f"Constraint: {constraint.compute(ans[:-1])}")
        print()


def gradient(func: Callable[[list[float]], float], point: list[float]):
    return approx_fprime(point, func, 1e-6)


def normalize(vector: np.ndarray) -> np.ndarray:
    ln = sqrt(sum(map(lambda x: x ** 2, vector)))
    return vector / ln


def gradient_descend(f: Function, constraint: Function, start: list[float], eps: float) -> tuple[list[float], int]:
    iterations = 0
    xn = start.copy()

    def norm_grad(point: list[float]):
        return sqrt(sum(map(lambda x: x ** 2, f.gradient(point))))

    params = AdamParams()
    while True:
        iterations += 1

        grad = gradient(norm_grad, xn)
        next_x = list(update_adam(params, np.asarray(xn), np.asarray(grad)))

        if abs(norm_grad(next_x)) < eps:  # and abs(constraint.compute(next_x[:-1])) < eps:
            return next_x, iterations

        xn = next_x.copy()

        # if iterations % 10_000 == 0:
        #     print(f"Iteration {iterations}. Current point is - {xn}")
        #     print(f.gradient(xn))
        #     print(f"Current gradient (direction of descent) - {grad}, it's norm diff = {abs(norm_grad(xn))}")
        #     print(f"Constraint: {constraint.compute(next_x[:-1])}")
        #     print()


if __name__ == "__main__":
    main()
