from typing import Callable

from Function import Function
from math import *
import numpy as np
from scipy.optimize import approx_fprime


def main():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 - z * (y + x**2)")
    starting_points = [
        [-0.050771933441697, 0.04868034322048902, 3.0003078438913837],
        [-0.01, 0.01, -0.5],
        [1, -1, 1.2],
        [0.5, 0.5, -0.7]
    ]
    for start in starting_points:
        result = gradient_descend(rosenbrok_lagrange, start, 1e-5)
        ans = result[0]
        value = rosenbrok_lagrange.compute(ans)
        print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")


def gradient(func: Callable[[list[float]], float], point: list[float]):
    return approx_fprime(point, func, 1e-6)


def check_wolfe_conditions(alpha: float, x: list[float], f: Callable[[list[float]], float], direction: np.ndarray[np.float64]):
    c1 = 0.025
    c2 = 0.2
    next = list(np.asarray(x, dtype='float64') + alpha * direction)
    f_next = f(next)
    fx = f(x)
    b = np.dot(gradient(f, x), direction)
    armijo_rule = f_next <= fx + alpha * c1 * b

    next_grad = gradient(f, next)
    lhs = np.dot(next_grad, direction)
    rhs = c2 * b
    curvature = lhs >= rhs
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

    while iterations < 50:
        iterations += 1

        if check_wolfe_conditions(step, current_x, f, direction):
            return step

        step *= coefficient
    return step


def normalize(vector: np.ndarray) -> np.ndarray:
    ln = sqrt(sum(map(lambda x: x**2, vector)))
    return vector / ln


def get_golden_ratio_step(f: Callable[[list[float]], float], xn: list[float]) -> float:
    phi = (1 + 5 ** 2) / 2
    a = 0
    b = 1
    grad = gradient(f, xn)
    mleft = b - (b - a) / phi
    mright = a + (b - a) / phi
    xl = list(np.asarray(xn) - mleft * grad)
    xr = list(np.asarray(xn) - mright * grad)
    lval = f(xl)
    rval = f(xr)

    while b - a > 1e-5:
        if lval < rval:
            b = mright
            mright = mleft
            mleft = b - (b - a) / phi
            rval = lval
            xr = xl
            xl = list(np.asarray(xn) - mleft * grad)
            lval = f(xl)
        else:
            a = mleft
            mleft = mright
            mright = a + (b - a) / phi
            lval = rval
            xl = xr
            xl = list(np.asarray(xn) - mright * grad)
            rval = f(xr)
    return a


def gradient_descend(f: Function, start: list[float], eps: float) -> tuple[list[float], int]:
    iterations = 0
    xn = start.copy()

    def norm_grad(point: list[float]):
        return np.linalg.norm(f.gradient(point))

    while True:
        iterations += 1

        grad = normalize(gradient(norm_grad, xn))
        grad[-1] *= -1
        # step = choose_step(xn, grad, norm_grad)
        step = get_golden_ratio_step(norm_grad, xn)
        next_x = list(np.asarray(xn) - step * grad)
        #
        # print(grad)
        # print(step)
        # print(next_x)
        # return None

        if abs(norm_grad(next_x) - norm_grad(xn)) < eps:
            return next_x, iterations

        xn = next_x.copy()

        # if iterations % 2000 == 0:
        print(f"Iteration {iterations}. Current point is - {xn}, step = {step}")


if __name__ == "__main__":
    main()