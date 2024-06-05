from scipy.optimize import approx_fprime
from sympy import hessian, symbols, sympify
from math import *


def count_variables(s: str) -> int:
    variables = ['x', 'y', 'z']
    was = set()
    for c in s:
        if c in variables:
            was.add(c)
    return len(was)


class Function:
    def __init__(self, func: str):
        self.string_value = func
        self.dimension: int = count_variables(func)
        if self.dimension not in [2, 3]:
            raise RuntimeError(f"Invalid dimension of a function's domain. Must be 2 or 3, got {self.dimension}")

    def __str__(self):
        return self.string_value

    def _get_func(self):
        def _func(X: list[float]) -> float:
            x = X[0]
            y = X[1]
            if self.dimension == 3:
                z = X[2]
            return eval(self.string_value)

        return _func

    def compute(self, point: list[float]) -> float:
        x = point[0]
        y = point[1]
        if self.dimension == 3:
            z = point[2]
        return eval(self.string_value)

    def gradient(self, point: list[float]):
        return approx_fprime(point, self._get_func(), 1e-6)

    def hessian(self):
        if self.dimension == 2:
            x, y = symbols('x y')
            func = sympify(self.string_value)
            return hessian(func, [x, y])
        else:
            x, y, z = symbols('x y z')
            func = sympify(self.string_value)
            return hessian(func, [x, y, z])

    def hessian_at_point(self, point: list[float]):
        if self.dimension == 2:
            x, y = symbols('x y')
            func = sympify(self.string_value)
            H = hessian(func, [x, y]).subs(x, point[0]).subs(y, point[1])
            return H
        x, y, z = symbols('x y z')
        func = sympify(self.string_value)
        H = hessian(func, [x, y]).subs(x, point[0]).subs(y, point[1]).subs(z, point[2])
        return H
