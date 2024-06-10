import random
import sys
from functools import reduce
import operator

import numpy as np
from scipy.optimize import approx_fprime


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


def get_V_i(N: int,
            i: int,
            s: list[np.ndarray],
            y: list[np.ndarray],
            rhos: list[float]) -> np.matrix:
    I = np.identity(N)
    return I - rhos[i] * get_vertical(y[i]) * s[i]


def get_steplenth(f: ParametrizedFunction, start: np.ndarray, direction: np.ndarray):
    e1 = 0.8
    e2 = 0.2
    l = 0.8
    a = 1
    f_last = f.compute(list(start))

    x_new = [x + a * p_i for x, p_i in zip(start, direction)]
    diff1 = f.compute(x_new) - f_last

    gradient_x_new = f.gradient(x_new)
    diff2 = np.dot(np.squeeze(np.asarray(gradient_x_new)), np.squeeze(np.asarray(direction)))

    gradient = f.gradient(list(start))
    tangent = np.dot(np.squeeze(np.asarray(gradient)), np.squeeze(np.asarray(direction)))
    l1 = e1 * a * tangent
    l2 = e2 * tangent

    eps = 1e-6
    while (diff1 - l1 > eps) and (diff2 - l2 >= eps):
        a = l * a

        x_new = [x + a * p_i for x, p_i in zip(start, direction)]
        diff1 = f.compute(x_new) - f_last

        gradient_x_new = f.gradient(x_new)
        diff2 = np.dot(np.squeeze(np.asarray(gradient_x_new)), np.squeeze(np.asarray(direction)))

        tangent = np.dot(np.squeeze(np.asarray(gradient)), np.squeeze(np.asarray(direction)))
        l1 = e1 * a * tangent
        l2 = e2 * tangent
    return a


def take_step(x: np.ndarray,
              function: ParametrizedFunction,
              H: np.matrix,
              s: list[np.ndarray],
              y: list[np.ndarray],
              rhos: list[float],
              k: int,
              m: int) -> tuple[np.ndarray, np.matrix]:
    grad = function.gradient(list(x))
    d = -np.asarray(H * get_vertical(grad)).transpose()[0]
    step = get_steplenth(function, x, d)
    x_next = x + step * d

    if len(s) == m:
        s.pop(0)
    new_s = x_next - x
    s.append(new_s)

    if len(y) == m:
        y.pop(0)
    new_y = function.gradient(list(x_next)) - function.gradient(list(x))
    y.append(new_y)

    if len(rhos) == m:
        rhos.pop(0)
    rho = 1 / (y[-1] * get_vertical(s[-1]))[0, 0]
    rhos.append(rho)

    m1 = min(k, m - 1)
    N = function.get_N()

    vs = [get_V_i(N, j, s, y, rhos).transpose() for j in range(m1, -1, -1)]
    left = reduce(operator.mul, vs, np.identity(N))
    vs_right = [get_V_i(N, j, s, y, rhos) for j in range(0, m1 + 1)]
    right = reduce(operator.mul, vs_right, np.identity(N))
    H_new = left * H * right

    for i in range(0, m1 + 1):
        center = get_vertical(s[i]) * s[i]
        vs = [get_V_i(N, j, s, y, rhos).transpose() for j in range(m1, i, -1)]
        left = reduce(operator.mul, vs, np.identity(N))
        vs_right = [get_V_i(N, j, s, y, rhos) for j in range(i + 1, m1 + 1)]
        right = reduce(operator.mul, vs_right, np.identity(N))
        H_new += rhos[i] * left * center * right
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

        if np.linalg.norm(function.gradient(list(x_next))) < eps:
            print(f"Finished in {k - m} iterations")
            return x_next if function.compute(list(x_next)) < function.compute(list(xn)) else xn

        k += 1
        xn = x_next.copy()


def test(m: int = 10):
    Ns = [2, 4, 6]
    for N in Ns:
        starting_point = np.random.rand(N) * random.randint(-5, 5)
        function = ParametrizedFunction(N)
        argmin = L_BFGS(list(starting_point), function, m)
        print("==== Arguments ====")
        print(f"N = {N}, m = {m}")
        print(f"Starting point = {starting_point}")
        print(f"Minimal value is {function.compute(list(argmin))} at {argmin}")
        print("===================")
        print()


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