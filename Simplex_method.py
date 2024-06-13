from typing import IO

import numpy as np


def get_printable_statement(coefficients: np.ndarray) -> list:
    tokens = []
    for idx, c in enumerate(coefficients):
        var = f"*x{idx + 1}"
        if idx == 0:
            tokens.append(str(c) + var)
        else:
            sign = "+" if c >= 0 else "-"
            tokens.append(sign)
            tokens.append(str(abs(c)) + var)
    return tokens


class LinearProgrammingProblem:
    """
    Class containing linear programming problem

    Fields:
        n - amount of variables\n
        m - amount of constraints\n
        statement - statement to minimize\n
        A - n*m matrix of constraints coefficients\n
        b - m*1 column of constraints values\n
    """
    def __init__(self, filename: str):
        self.n: int = -1
        self.m: int = -1
        self.statement: np.ndarray = None
        self.A: np.matrix = None
        self.b: np.ndarray = None
        file = open(filename, 'r')
        self.initialize(file)
        file.close()

    def initialize(self, file: IO):
        lines = file.readlines()
        self.n, self.m = map(int, lines[0].split())
        self.statement = np.asarray(list(map(int, lines[1].split())))
        mx = []
        b = []
        for i in range(2, len(lines)):
            row = list(map(int, lines[i].split()))
            b.append(row[-1])
            mx.append(row[:-1])
        self.A = np.asmatrix(mx)
        self.b = np.asarray(b)

    def __str__(self):
        res = ""
        res += "Linear Programming Problem:\n"
        res += "===========================\n"
        res += "Minimize:\n"
        tokens = get_printable_statement(self.statement)
        res += " ".join(tokens) + "\n"

        res += "With constraints:\n"
        for row, b_i in zip(self.A, self.b):
            tokens = get_printable_statement(np.asarray(row)[0])
            res += " ".join(tokens) + " <= " + str(b_i) + "\n"
        return res


def get_vertical(arr: np.ndarray) -> np.ndarray:
    return np.asmatrix(arr).transpose()


def preparations(problem: LinearProgrammingProblem) -> np.ndarray:
    last_row = np.concatenate((-problem.statement, np.zeros(problem.n + problem.m + 1 - len(problem.statement))))
    matrix = np.vstack((np.hstack((problem.A, np.identity(problem.m), get_vertical(problem.b))), last_row))
    return matrix


def should_terminate(matrix: np.ndarray) -> bool:
    last_row = np.asarray(matrix[-1, :-1])[0]
    return all(list(map(lambda x: int(x <= 0), last_row)))


def make_iteration(m: int, matrix: np.ndarray, basis: list[int]) -> tuple[np.ndarray, list]:
    idx = 0
    for i in range(len(matrix)):
        if matrix[-1, i] >= matrix[-1, idx]:
            idx = i

    d = 0
    mn = 10 ** 18
    for i in range(m):
        if matrix[i, idx] <= 0:
            continue
        if mn > matrix[i, -1] / matrix[i, idx]:
            d = i
            mn = matrix[i, -1] / matrix[i, idx]

    basis[d] = idx
    value = matrix[d, idx]
    matrix[d, :] /= value
    for i in range(len(matrix)):
        if i == d:
            continue
        val = matrix[i, idx]
        for j in range(matrix.shape[1]):
            matrix[i, j] -= matrix[d, j] * val
    return matrix, basis


def get_answer(matrix: np.ndarray, basis: list[int]) -> list[float]:
    ans = [0 for _ in range(matrix.shape[1])]
    for i in range(len(basis)):
        ans[basis[i]] = matrix[i, -1]
    return ans


def simplex(problem: LinearProgrammingProblem) -> tuple[list[float], float]:
    matrix = preparations(problem)

    iterations = 0
    basis = [problem.n + i for i in range(problem.m)]
    while not should_terminate(matrix):
        iterations += 1

        matrix, basis = make_iteration(problem.m, matrix, basis)

    ans = get_answer(matrix, basis)
    value = matrix[-1, -1]
    return ans, value


def main():
    problem = LinearProgrammingProblem("linear_problem.txt")
    print(problem)

    res, value = simplex(problem)
    print(f"Minimal value is {value}")
    print("Values of variables are as follows:")
    for i in range(problem.n):
        print(f"x{i + 1} = {res[i]}")


if __name__ == "__main__":
    main()
