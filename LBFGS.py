import sys

import numpy as np
from scipy.optimize import approx_fprime


class ParametrizedFunction:
    def __init__(self, parameter: int):
        self._N = parameter

    def get_N(self):
        return self._N

    def compute(self, point: list[float]):
        values = [100 * (point[2 * idx - 2] ** 2 - point[idx * 2 - 1]) + (point[idx * 2 - 2] - 1) ** 2
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


def take_step(s: list[np.ndarray],
              y: list[np.ndarray],
              rhos: list[float],
              function: ParametrizedFunction,
              start: list[float],
              m: int) -> np.ndarray:
    q = function.gradient(start)
    alphas = []
    for i in range(m - 1, -1, -1):
        alpha = rhos[i] * s[i] * get_vertical(q)
        alphas.append(alpha)
        q -= alpha * y[i]

    gamma = (s[-1] * get_vertical(y[-1])) / (y[-1] * get_vertical(y[-1]))
    H = gamma * np.ones((function.get_N(), function.get_N()))
    z = H * q

    for i in range(m):
        beta = rhos[i] * y[i] * z
        z += get_vertical(s[i]) * (alphas[i] - beta)
    return z


def L_BGFGS(start: list[float], function: ParametrizedFunction, m: int, eps: float = 1e-6) -> np.ndarray:
    values = get_initial_values(start, function)
    s = [values[0]]
    y = [values[1]]
    rhos = [1 / np.dot(s[0], y[0])]
    N = function.get_N()

    # print(np.asmatrix(y[0]) * get_vertical(s[0])) # Returns number
    # print(get_vertical(s[0]) * np.asmatrix(y[0])) # Returns matrix

    # Filling history
    for i in range(1, m):
        s.append(s[-1])
        y.append(y[-1])
        rhos.append(rhos[-1])

    xn = np.asarray(start.copy())
    while True:
        direction = take_step(s, y, rhos, function, xn, m)
        x_next = xn + direction

        if abs(function.compute(list(x_next)) - function.compute(list(xn))) < eps:
            return x_next

        s.pop(0)
        s.append(x_next - xn)

        grad = function.gradient(xn)
        grad_next = function.gradient(x_next)
        y.pop(0)
        y.append(grad_next - grad)

        rho_next = 1 / (y[-1] * get_vertical(s[-1]))
        rhos.pop(0)
        rhos.append(rho_next)


def main():
    print(sys.argv)
    N = int(sys.argv[1])
    function = ParametrizedFunction(N)
    print(function.compute([2, 1]))


if __name__ == "__main__":
    """
    idk why it didn't work from console, so to use command line argument for parameter N
    I had to edit run configuration in PyCharm
    """
    main()