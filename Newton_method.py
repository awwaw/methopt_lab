import numpy as np

from Function import Function
from numpy.linalg import LinAlgError
import math


def main():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 - z * (y + x**2)")
    rosenbrok = Function("(1-x)**2 + 100 * (y-x**2)**2")
    # xs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # ys = [-0.02, -0.019, -0.018, -0.017, -0.016]
    # zs = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
    # epss = [1e-8, 1e-7, 1e-9, 1e-6, 1e-5]
    # results = []
    # for x in xs:
    #     for y in ys:
    #         for z in zs:
    #             for eps in epss:
    #                 print(f"Starting with x = {x}, y = {y}, lambda = {z}, eps = {1e-5}")
    #                 res = newton(rosenbrok_lagrange, [x, y, z], 10**-5)
    #                 point = res[0]
    #                 value = rosenbrok.compute(point)
    #                 iterations = res[1]
    #                 results.append((value, iterations, point, [x, y, z, 10**-5]))
    # results.sort()
    # best_val = results[0][0]
    # best_iters = results[0][1]
    # best_point = results[0][2]
    # print(f"Found minimum of function in {best_iters} iterations.\n Value is {best_val} at {best_point}")
    # print(f"Best args - {results[0][-1]}")
    start = [0.15, -0.0225, -3]
    result = newton(rosenbrok_lagrange, start, 1e-5)
    ans = result[0]
    value = rosenbrok.compute(ans)
    print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")


def norm(x: list[float]) -> float:
    """
    Idk why np.linalg.norm didn't work, so I made this xd

    :param x:
    :return:
    """
    return math.sqrt(sum(map(lambda x_: x_ ** 2, x)))


def stop_condition(eps: float, last_x: list[float], current_x: list[float], next_x: list[float]) -> bool:
    """
    Checks if Newton's method should stop.
    Used formula is stated in the page below:
    https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9D%D1%8C%D1%8E%D1%82%D0%BE%D0%BD%D0%B0#%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC

    :param eps: a value of "absolute error" since this method is an approximate
    :param last_x: previous point
    :param current_x: current point
    :param next_x: next point
    :return: true if algorithm must stop, else false
    """

    if last_x is None:
        return False
    last = np.asarray(last_x)
    cur = np.asarray(current_x)
    next = np.asarray(next_x)
    a = norm(next - cur)
    b = norm(cur - last)
    return a / abs(1 - a / b) < eps


def newton(f: Function, point: list[float], epsilon: float) -> tuple[list[float], int]:
    """
    Constrained Newton's method implication.

    :param f: Lagrangian of a function which minimization is desired
    :param point: starting approximation of minimum
    :param epsilon: value of approximation
    :return: tuple of resulting point and number of iterations
    """
    last_x = None
    xn = point
    iterations = 0

    while True:
        iterations += 1
        # inversed_hessian = f.hessian_at_point(xn).inv()
        # gradient = f.gradient(point)
        # direction = np.asarray(inversed_hessian * gradient)[0]
        #
        # next_x = list(np.asarray(xn) - direction)
        hessian = f.hessian_at_point(xn)
        if hessian.det() == 0.0:
            raise LinAlgError(f"det of Hessian of function at point {xn} is equal to 0")

        gradient = f.gradient(xn)
        p = np.asarray(-(hessian.inv() * gradient))
        next_x = [x_i + p_i[0] for x_i, p_i in zip(xn, p)]

        if stop_condition(epsilon, last_x, xn, next_x):
            return next_x, iterations

        last_x = xn.copy()
        xn = next_x.copy()

        if iterations % 2_000 == 0:
            print(f"Iteration {iterations}")
            print(f"Current point is {xn}")


if __name__ == "__main__":
    main()
