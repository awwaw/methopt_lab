import numpy as np

from Function import Function
from numpy.linalg import LinAlgError
from math import *


def main():
    rosenbrok = Function("(1-x)**2 + 100 * (y-x**2)**2")
    constraint = Function("y + x**2")
    starting_points = [
        [0.15, -0.0225],
        [-0.01, 0.01],
        [1, -1],
        [0.5, 0.5]
    ]
    for start in starting_points:
        result = newton(rosenbrok, constraint, start, 1e-5)
        ans = result[0]
        value = rosenbrok.compute(ans)
        print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")


def norm(x: list[float]) -> float:
    """
    Idk why np.linalg.norm didn't work, so I made this xd

    :param x:
    :return:
    """
    return sqrt(sum(map(lambda x_: x_ ** 2, x)))


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


def compute_KKT(f: Function, constraint: Function, point: list[float]) -> np.ndarray:
    """
    Method used to find dx in Newton's method.\n
    Here we solve system of equations with KKT matrix made of function's Hessian and constraint's gradient

    :param f: function to minimize
    :param constraint: well, constraint
    :param point: current point
    :return: step that we need to do to progress in newton's method
    """
    hessian = f.hessian_at_point(point)
    if hessian.det() == 0.0:
        raise LinAlgError(f"det of Hessian of function at point {point} is equal to 0")

    gradient = f.gradient(point)
    constraint_gradient = constraint.gradient(point)

    KKT = np.matrix(hessian, dtype='float64')
    KKT = np.vstack((KKT, constraint_gradient))
    column = np.append(constraint_gradient, [0.]).reshape((3, 1))
    KKT = np.append(KKT, column, axis=1)

    constraint_value = constraint.compute(point)
    b_column = np.append(gradient, constraint_value)
    b_column = (b_column * (-1)).reshape((3, 1))

    # Now we have to solve equation KKT * x = b to find our step

    X = np.linalg.solve(KKT, b_column).reshape((1, 3))[0]
    return X


def newton(f: Function, h: Function, point: list[float], epsilon: float) -> tuple[list[float], int]:
    """
    Constrained Newton's method implication.

    :param f: Function minimization of which is desired
    :param h: Constraint function
    :param point: starting approximation of minimum
    :param epsilon: value of approximation
    :return: tuple of resulting point and number of iterations
    """
    last_x = None
    xn = point
    iterations = 0

    while True:
        iterations += 1

        X = compute_KKT(f, h, xn)
        dx = X[0]
        dy = X[1]
        dl = X[2]

        next_x = list(np.asarray(xn) + np.asarray([dx, dy]))

        if stop_condition(epsilon, last_x, xn, next_x):
            return next_x, iterations

        last_x = xn.copy()
        xn = next_x.copy()

        if iterations % 2_000 == 0:
            print(f"Iteration {iterations}")
            print(f"Current point is {xn}")


def test_1():
    paraboloid = Function("x**2 / 2 + y**2 / 2")
    constraint = Function("x + y")
    start = [0.15, -0.0225]
    result = newton(paraboloid, constraint, start, 1e-5)
    ans = result[0]
    value = paraboloid.compute(ans)
    print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")


def test_2():
    paraboloid = Function("-20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))")
    constraint = Function("y**2 + x**2 - 9")
    start = [-0.06, 0.25]
    result = newton(paraboloid, constraint, start, 1e-5)
    ans = result[0]
    value = paraboloid.compute(ans)
    print(f"Found minimum in {result[-1]} iterations. Minimum value is {value} at {ans}")


def test():
    main()
    print()
    # test_1()
    # print()
    # test_2()


if __name__ == "__main__":
    test()
