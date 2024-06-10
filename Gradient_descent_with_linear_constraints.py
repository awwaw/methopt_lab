import numpy as np
from Function import Function
from scipy.linalg import null_space


class Constraints:
    def __init__(self, A: np.matrix, b: np.matrix):
        self.A = A
        self.b = b

    def kernel(self) -> np.matrix:
        ker = null_space(self.A)
        return np.asmatrix(ker)


def check_for_system_of_equations(lines: list[str]) -> bool:
    for line in lines:
        if "=" in line:
            return True
    return False


def get_variable_index(variable: str) -> int:
    pos = variable.index('x')
    try:
        idx = int(variable[pos + 1:])
    except ValueError:
        raise RuntimeError("Incorrect input format: variables must look like x[index]")
    return idx


def parse_equation(equation: str, dimension: int) -> list[float]:
    tokens = equation.split()
    coefficients: list[float] = [0 for _ in range(dimension)]
    minus = 1
    for token in tokens:
        if 'x' not in token:
            if token == '-':
                minus = -1
            continue
        # Here we guarantee that there is 'x' in this token
        pos = token.index('x')
        idx = get_variable_index(token) - 1
        if pos == 0:
            coefficients[idx] = 1.0 * minus
        else:
            try:
                coeff = float(token[:pos])
                coefficients[idx] = coeff * minus
            except ValueError:
                raise RuntimeError("Incorrect input format")
        minus = 1
    return coefficients


def get_dimension(lines: list[str]) -> int:
    dim = 0
    for line in lines:
        tokens = line.split()
        for token in tokens:
            if 'x' in token:
                dim = max(dim, get_variable_index(token))
    return dim


def parse_system(lines: list[str]) -> tuple[np.matrix, np.matrix]:
    A = []
    b = []
    dimension = get_dimension(lines)
    for line in lines:
        left, right = list(map(lambda x: x.strip(), line.split("=")))
        b.append([float(right)])
        A.append(parse_equation(left, dimension))
    return np.asmatrix(A), np.asmatrix(b)


def parse_matrices(lines: list[str]) -> tuple[np.matrix, np.matrix]:
    A = []
    b = []
    for line in lines:
        left, right = list(map(lambda x: x.strip(), line.split("|")))
        A.append(list(map(float, left.split())))
        b.append([float(right)])
    if len(A) != len(b):
        raise RuntimeError("Amount of rows in left and right matrices must be equal")
    return np.asmatrix(A), np.asmatrix(b)


def parse_constraints() -> Constraints:
    A: np.matrix = np.matrix([])
    b: np.matrix = np.matrix([])
    with open("constraints.txt", "r") as constraints_file:
        lines = constraints_file.readlines()
        if check_for_system_of_equations(lines):
            A, b = parse_system(lines)
        else:
            A, b = parse_matrices(lines)
    return Constraints(A, b)


def get_starting_point(constraints: Constraints) -> np.ndarray:
    """
    Here we get our starting point x for gradient descent such that Ax = b.\n
    This solution is approximate, because np.linalg.norm requires square matrix which we don't always have.
    So instead I'm using np.linalg.lstsq, which finds arg min ||Ax - b||^2

    :param constraints: constraints for a function
    :return: starting point satisfying constraints
    """

    x = np.linalg.lstsq(constraints.A, constraints.b, rcond=None)[0]
    return np.asarray(x)


def get_projection(vector: np.ndarray, subspace_basis: np.matrix) -> np.ndarray:
    """
    Getting a projection of a vector onto a subspace

    :param vector: vector to be projected
    :param subspace_basis: orthonormal basis of a subspace to project vector on
    :return: projection of a given vector onto subspace
    """

    y = np.zeros(vector.shape)
    for r in subspace_basis.transpose():
        row = np.asarray(r)[0]
        y += np.dot(vector, row) * row
    return y


def gradient_descend(starting_point: np.ndarray,
                     function: Function,
                     kernel: np.matrix,
                     step: float) -> tuple[np.ndarray, int]:
    xn = starting_point
    iterations = 0

    while True:
        iterations += 1
        grad = function.gradient(list(xn))
        direction = get_projection(grad, kernel)

        x_next = xn - step * direction

        # if abs(function.compute(list(x_next)) - function.compute(list(xn))) < 1e-8:
        if np.linalg.norm(direction) < 1e-6:
            return x_next, iterations

        xn = x_next


def main():
    constraints = parse_constraints()
    x = get_starting_point(constraints)
    print(np.allclose(constraints.A * x, constraints.b))
    x = x.transpose()[0]
    ker = constraints.kernel()
    # function = Function("(1 - x)**2 + 100 * (y - x**2)**2")
    function = Function("x**2 + y**2 + z**2")
    result, iterations = gradient_descend(x, function, ker, 1e-5)
    print(f"Found minimum in {iterations} iterations")
    print(f"Minimum of function is {function.compute(list(result))} at {result}")


if __name__ == "__main__":
    main()
