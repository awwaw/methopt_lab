import random
import sys
import time
from functools import reduce
import operator

import numpy as np
from scipy.optimize import approx_fprime
from Gradient_descent import choose_step


class ParametrizedFunction:
    """
    Class for given function, which can compute value of a function and gradient at given point
    """
    def __init__(self, parameter: int):
        self._N = parameter

    def get_N(self):
        return self._N

    def compute(self, point: list[float]):
        values = [100 * (point[2 * idx - 2] ** 2 - point[idx * 2 - 1]) ** 2 + (point[idx * 2 - 2] - 1) ** 2
                  for idx in range(1, self._N // 2 + 1)]
        return sum(values)

    def gradient(self, point: list[float]):
        return approx_fprime(point, self.compute, 1e-6)


def get_initial_values(start: list[float], function: ParametrizedFunction) -> list[np.ndarray]:
    x0 = start
    y0 = function.gradient(x0)
    step = 2e-5
    x1 = list(np.asarray(x0) - step * np.asarray(y0))
    y1 = function.gradient(x1)
    return [np.asarray(x1) - np.asarray(x0), np.asarray(y1) - np.asarray(y0)]


def get_vertical(arr: np.ndarray) -> np.ndarray:
    return np.asmatrix(arr).transpose()


def get_index(k: int, m: int, i: int) -> int:
    """
    [k - m + 1, k] -> [0, m - 1]

    :param k:
    :param m:
    :param i:
    :return:
    """
    if k < m:
        return i - (k - m + 1)
    return i - 1


def get_V_i(N: int,
            i: int,
            s: list[np.ndarray],
            y: list[np.ndarray],
            rhos: list[float]) -> np.matrix:
    I = np.identity(N)
    if len(s) > 0:
        return I - rhos[i - 1] * get_vertical(y[i - 1]) * s[i - 1]
    return np.asmatrix(I)


def take_step(x: np.ndarray,
              function: ParametrizedFunction,
              H: np.matrix,
              s: list[np.ndarray],
              y: list[np.ndarray],
              rhos: list[float],
              k: int,
              m: int) -> tuple[np.ndarray, np.matrix]:
    grad = function.gradient(list(x))
    d = np.asarray(-H * get_vertical(grad)).transpose()[0]
    print(f"grad = {grad}")
    print(f"H = {H}")
    print(f"x = {x}")
    print(f"d = {d}")
    step = choose_step(list(x), grad, function.compute)
    x_next = x + step * d

    m1 = min(k, m - 1)
    N = function.get_N()
    # V = get_V_i(N, 0, s, y, rhos).transpose()
    # for i in range(1, m1 + 1):
    #     print(k, m1, i, k - i, k - m1)
    #     V_i = get_V_i(N, k - i, s, y, rhos).transpose()
    #     V *= V_i
    # V_right = get_V_i(N, 1, s, y, rhos)
    # for i in range(1, k + 1):
    #     V_i = get_V_i(N, i, s, y, rhos)
    #     V_right *= V_i
    vs = [get_V_i(N, j, s, y, rhos).transpose() for j in range(m1, -1, -1)]
    left = reduce(operator.mul, vs, np.identity(N))
    vs_right = [get_V_i(N, j, s, y, rhos) for j in range(1, m1 + 1)]
    right = reduce(operator.mul, vs_right, np.identity(N))
    H_new = left * H * right

    # for i in range(k - m1, k + 1):
    for i in range(0, m1 + 1):
        if len(s) > 0:
            center = get_vertical(s[i - 1]) * s[i - 1]
            vs = [get_V_i(N, j, s, y, rhos).transpose() for j in range(m1, -1, -1)]
            left = reduce(operator.mul, vs, np.ones((N, N)))
            vs_right = [get_V_i(N, j, s, y, rhos) for j in range(1, m1 + 1)]
            right = reduce(operator.mul, vs_right, np.ones((N, N)))
            H_new += rhos[i - 1] * left * center * right
        else:
            H_new = H
    return x_next, H_new


def L_BFGS(start: list[float], function: ParametrizedFunction, m: int, eps: float = 1e-6) -> np.ndarray:
    N = function.get_N()
    H = np.identity(N)
    s = []
    y = []
    rhos = []
    xn = np.asarray(start)

    k = 0
    while True:
        x_next, H_next = take_step(xn, function, H, s, y, rhos, k, m)
        H = H_next

        print(x_next)
        print(H_next)

        if len(s) == m:
            s.pop(0)
        new_s = x_next - xn
        s.append(new_s)

        if len(y) == m:
            y.pop(0)
        new_y = function.gradient(list(x_next)) - function.gradient(list(xn))
        y.append(new_y)

        if len(rhos) == m:
            rhos.pop(0)
            rho = 1 / (y[-1] * get_vertical(s[-1]))[0, 0]
            # rhos.append(rho)
            rhos.append(1)
        else:
            rhos.append(1)

        print("Result on this step")
        print(x_next)
        print(xn)
        print(function.compute(list(x_next)))
        print(function.compute(list(xn)))
        if k > m and abs(function.compute(list(x_next)) - function.compute(list(xn))) < eps:
            print(f"Finished in {k - m} iterations")
            return x_next if function.compute(list(x_next)) < function.compute(list(xn)) else xn

        k += 1


def test(m: int = 10):
    # Ns = [2, 4, 6]
    Ns = [2]
    for N in Ns:
        starting_point = np.random.rand(N) * random.randint(-5, 5)
        function = ParametrizedFunction(N)
        argmin = L_BFGS(list(starting_point), function, m)
        print("==== Arguments ====")
        print(f"N = {N}, m = {m}")
        print(f"Starting point = {starting_point}")
        print(f"Minimal value is {function.compute(list(argmin))} at {argmin}")
        print("===================")


def main():
    # N = int(sys.argv[1])
    # function = ParametrizedFunction(N)
    test()


if __name__ == "__main__":
    """
    idk why it didn't work from console, so to use command line argument for parameter N
    I had to edit run configuration in PyCharm
    """
    main()