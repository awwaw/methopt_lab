import numdifftools as nd
from scipy.optimize import approx_fprime
from sympy import hessian, symbols, sympify


class Function:
    def __init__(self, func: str):
        self.string_value = func

    def _get_func(self):
        def _func(X: list[float]) -> float:
            x = X[0]
            y = X[1]
            z = X[2]
            return eval(self.string_value)

        return _func

    def compute(self, point: list[float]) -> float:
        x = point[0]
        y = point[1]
        z = point[2]
        return eval(self.string_value)

    def gradient(self, point: list[float]):
        return approx_fprime(point, self._get_func(), 1e-6)

    def hessian(self):
        x, y, z = symbols('x y z')
        func = sympify(self.string_value)
        return hessian(func, [x, y, z])

    def hessian_at_point(self, point: list[float]):
        x, y, z = symbols('x y z')
        func = sympify(self.string_value)
        H = hessian(func, [x, y, z]).subs(x, point[0]).subs(y, point[1]).subs(z, point[2])
        return H
