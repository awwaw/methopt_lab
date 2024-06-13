from Simplex_method import *

eps = 1e-5


def get_mx(result: list[float]) -> int:
    mx = -1
    for i in range(len(result)):
        if mx == -1 and abs(result[i] - round(result[i])) > 1e-5:
            mx = i
        else:
            continue
        if result[mx] % 1 <= result[i] % 1:
            mx = i
    return mx


def get_new_basis_element(new_matrix: list[list[float]],
                          n: int,
                          m: int) -> int:
    idx = -1
    mx = -10 ** 18
    for i in range(len(new_matrix[0]) - 1):
        if abs(new_matrix[-2][i]) > 1e-12 and i != n + m:
            val = new_matrix[-1][i] / new_matrix[-2][i]
            if val > mx:
                mx = val
                idx = i
    return idx


def make_zeros(new_matrix: list[list[float]],
               idx: int,
               m: int) -> list[list[float]]:
    value = new_matrix[m][idx]
    for i in range(len(new_matrix[m])):
        new_matrix[m][i] /= value

    for i in range(len(new_matrix)):
        if i != m:
            v = new_matrix[i][idx]
            for j in range(len(new_matrix[i])):
                new_matrix[i][j] -= v * new_matrix[m][j]
            new_matrix[i][idx] = 0
    return new_matrix


def initialize_new_matrix(matrix: list[list[float]],
                          row_idx: int,
                          result: list[float],
                          mx: int) -> list[list[float]]:
    new_row = []
    for j in range(len(matrix[row_idx]) - 1):
        val = matrix[row_idx][j]
        new_row.append(-(val % 1))
    new_row.append(1)
    new_row.append(-(result[mx] % 1))

    new_matrix = []
    for i in range(len(matrix) - 1):
        row = matrix[i]
        new_matrix.append(row[:-1] + [0] + [row[-1]])
    new_matrix.append(new_row)
    new_matrix.append(matrix[-1][:-1] + [0] + [matrix[-1][-1]])
    return new_matrix


def gomory(matrix: list[list[float]],
           result: list[float],
           basis: list[int]) -> tuple[list, list[float], list[int]]:
    m = len(matrix) - 1
    n = len(matrix[0]) - m - 1

    mx = get_mx(result)
    if mx == -1:
        return matrix, result, basis

    row_idx = basis.index(mx) if mx in basis else 0
    new_matrix = initialize_new_matrix(matrix, row_idx, result, mx)

    idx = get_new_basis_element(new_matrix, n, m)
    basis.append(idx)

    new_matrix = make_zeros(new_matrix, idx, m)

    new_result = get_answer(np.asmatrix(new_matrix), basis)
    return gomory(new_matrix, new_result, basis)


def main():
    problem = LinearProgrammingProblem("linear_problem.txt", True)
    matrix, basis = simplex(problem, True)
    matrix[-1, :] *= -1
    result = get_answer(matrix, basis)

    m = []
    for row in matrix:
        m.append(list(np.asarray(row)[0]))
    matrix, result, basis = gomory(m, result, basis)

    print(problem)
    print(f"Minimal value is {-matrix[-1][-1]}")
    print("Values of variables are as follows:")
    for i in range(problem.n):
        print(f"x{i + 1} = {round(result[i])}")


if __name__ == "__main__":
    main()
