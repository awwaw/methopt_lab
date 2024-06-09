from typing import Callable

from Function import Function
from math import *
import numpy as np
from scipy.optimize import approx_fprime


def main():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 - z * (y + x**2)")
    constraint = Function("y + x**2")

    """
    Here we can see, that first 4 starting points end up with arguably good minimum value, but 5th is going bad :(
    """
    starting_points = [
        [0.103, -0.01, 0],
        [0.05, -0.05, 0],
        [-0.01, 0.01, 0],
        [1, -1, 1.2],
        [0.23, 0.32, 0]
    ]

    for start in starting_points:
        result = gradient_descend(rosenbrok_lagrange, constraint, start, 1e-4)
        ans = result[0]
        value = rosenbrok_lagrange.compute(ans)
        print()
        print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")
        print(f"Constraint: {constraint.compute(ans[:-1])}")
        print()


def gradient(func: Callable[[list[float]], float], point: list[float]):
    return approx_fprime(point, func, 1e-6)


def check_wolfe_conditions(alpha: float, x: list[float], f: Callable[[list[float]], float],
                           direction: np.ndarray[np.float64]):
    c1 = 0.2
    c2 = 0.8
    next = list(np.asarray(x, dtype='float64') + alpha * direction)
    f_next = f(next)
    fx = f(x)
    b = np.dot(gradient(f, x), direction)
    armijo_rule = f_next <= fx + alpha * c1 * b

    next_grad = gradient(f, next)
    lhs = np.dot(next_grad, direction)
    rhs = c2 * b
    curvature = lhs - rhs >= 1e-5
    return armijo_rule and curvature


def choose_step(current_x: list[float], direction: np.ndarray, f: Callable[[list[float]], float]) -> float:
    """
    Choose step for gradient descend using Wolfe conditions
    https://indrag49.github.io/Numerical-Optimization/line-search-descent-methods.html#the-wolfe-conditions
    Here I am using some dumb approach with linear search of step sice satisfying Wolfe conditions

    :param current_x:
    :param direction:
    :param f:
    :return:
    """

    step = 1.
    coefficient = 0.8
    iterations = 0

    while True:  # iterations < 50:
        iterations += 1

        if check_wolfe_conditions(step, current_x, f, direction):
            return step

        step *= coefficient


def normalize(vector: np.ndarray) -> np.ndarray:
    ln = sqrt(sum(map(lambda x: x ** 2, vector)))
    return vector / ln


def gradient_descend(f: Function, constraint: Function, start: list[float], eps: float) -> tuple[list[float], int]:
    iterations = 0
    xn = start.copy()

    def norm_grad(point: list[float]):
        return sqrt(sum(map(lambda x: x ** 2, f.gradient(point))))

    while True:
        iterations += 1

        grad = gradient(norm_grad, xn)
        step = choose_step(xn, grad, norm_grad)
        next_x = list(np.asarray(xn) - step * grad)

        if abs(norm_grad(next_x) - norm_grad(xn)) < eps:  # and abs(constraint.compute(next_x[:-1])) < eps:
            return next_x, iterations

        last_x = xn.copy()
        xn = next_x.copy()

        # if iterations % 20 == 0:
        #     print(f"Iteration {iterations}. Current point is - {xn}, step = {step}")
        #     print(f.gradient(xn))
        #     print(f"Current gradient (direction of descent) - {grad}, it's norm diff = {abs(norm_grad(xn) - norm_grad(last_x))}")
        #     print(f"Constraint: {constraint.compute(next_x[:-1])}")
        #     print()


if __name__ == "__main__":
    main()
