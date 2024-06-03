import numpy as np

from Function import Function
from numpy.linalg import LinAlgError
from scipy.optimize import brute


def main():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 + z * (y + x**2)")
    rosenbrok = Function("(1-x)**2 + 100 * (y-x**2)**2")
    # res = newton(rosenbrok_lagrange, [0.14, -0.018, -0.5], 1e-8, 1)
    xs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    ys = [-0.02, -0.019, -0.018, -0.017, -0.016]
    zs = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
    epss = [1e-8, 1e-7, 1e-9, 1e-6, 1e-5]
    results = []
    for x in xs:
        for y in ys:
            for z in zs:
                for eps in epss:
                    res = newton(rosenbrok_lagrange, [x, y, z], eps, 1)
                    point = res[0]
                    value = rosenbrok.compute(point)
                    iterations = res[1]
                    results.append((value, iterations, point, [x, y, z, eps]))
    results.sort()
    best_val = results[0][0]
    best_iters = results[0][1]
    best_point = results[0][2]
    print(f"Found minimum of function in {best_iters} iterations.\n Value is {best_val} at {best_point}")
    print(f"Best args - {results[0][-1]}")


def grid_search():
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 + z * (y + x**2)")




def get_score(point: list[float]) -> float:
    rosenbrok_lagrange = Function("(1-x)**2 + 100 * (y-x**2)**2 + z * (y + x**2)")
    return rosenbrok_lagrange.compute(point)


def newton(f: Function, x0: list[float], epsilon: float, step: float) -> tuple[list[float], int]:
    xn = x0
    iterations = 0

    while True:
        iterations += 1
        hessian = f.hessian_at_point(xn)
        if hessian.det() == 0.0:
            raise LinAlgError(f"Hessian of function at point {xn} is equal to 0")

        gradient = f.gradient(xn)
        p = np.asarray(-((hessian ** (-1)) * gradient))
        xnew = [x_i + step * p_i[0] for x_i, p_i in zip(xn, p)]

        if abs(f.compute(xnew) - f.compute(xn)) < epsilon:
            break

        xn = xnew.copy()
        if iterations % 2000 == 0:
            print(f"vector - {p}")
            print(f"Iteration - {iterations}, current point - {xn}")
    return xn, iterations


if __name__ == "__main__":
    main()