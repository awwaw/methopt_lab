import random
from typing import Callable

import numpy as np


def generate_graph(N: int, max_cost: int = 10) -> list[list[int]]:
    """
    Generates a graph with Hamiltonian cycle using Dirac's Theorem

    :param N: amount of vertices
    :param max_cost: maximal cost on the edge
    :return: adjacency matrix of result graph
    """

    matrix = [[0 for _ in range(N)] for __ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                matrix[i][j] = random.randint(1, max_cost)
    return matrix


def choose_edges(n: int) -> tuple[int, int]:
    """
    Get a pair of vertices in permutation which will be starting points of chosen edges

    :param n:
    :return:
    """

    while True:
        v, u = random.sample(range(0, n - 1), 2)
        if max(v, u) - min(v, u) <= 2:
            continue
        return v, u


def get_cost(graph: list[list[int]], permutation: np.ndarray) -> int:
    """
    Returns a summary cost of path in given graph

    :param graph:
    :param permutation:
    :return:
    """

    sm = 0
    for i in range(len(permutation) - 1):
        v, u = permutation[i], permutation[i + 1]
        sm += graph[v][u]
    sm += graph[permutation[0]][permutation[-1]]
    return sm


def genetic(graph: list[list[int]],
            N: int,
            M: int,
            mu: float,
            nu: float,
            epochs: int = 10_000) -> list[np.ndarray]:
    """
    Genetic algorythm for Traveling Salesman Problem

    :param graph: graph
    :param N: size of population
    :param M: size of mutating population
    :param mu: probability of mutation
    :param nu: probability of crossover
    :param epochs: amount of epochs of evolution (default = 20)
    :return: best solution
    """

    V = len(graph)
    population = [
        np.random.permutation(V) for _ in range(N)
    ]

    mn = 10**18
    for perm in population:
        cost = get_cost(graph, perm)
        mn = min(cost, mn)
    print(f"Min cost on random permutations = {mn}")
    print("====================")

    for epoch in range(epochs):
        # mutating permutations
        indices = random.choices(list(range(0, N)), k=M)
        for i in indices:
            check = np.random.rand()
            if check <= mu:
                v, u = choose_edges(len(population[i]))
                population[i][v], population[i][u] = population[i][u], population[i][v]

        best = []

        # crossovers
        for idx in indices:
            check = np.random.rand()
            if check <= nu:
                second = random.choice(indices)
                a, b = sorted(list(random.sample(range(0, len(population[0])), 2)))
                son = population[idx].copy()
                daughter = population[idx].copy()
                for i in range(a, b + 1):
                    son[i], daughter[i] = daughter[i], son[i]  # looks like incest
                son_cost = get_cost(graph, son)
                daughter_cost = get_cost(graph, daughter)
                if get_cost(graph, son) < get_cost(graph, daughter):
                    best.append((son_cost, son))
                else:
                    best.append((daughter_cost, daughter))

        for idx in indices:
            best.append((get_cost(graph, population[idx]), population[idx]))
        best = sorted(best, key=lambda x: x[0])[:N]
        new_population = [
            b[1] for b in best
        ]
        population = new_population.copy()
    return population


def main():
    graph = generate_graph(100)
    res = genetic(graph, 100, 150, 0.1, 0.1)
    mn = 10**18
    ans = []
    print("Best result")
    for r in res:
        cost = get_cost(graph, r)
        if cost < mn:
            mn = cost
            ans = r
    print(f"Permutation:\n {ans}\nCost = {mn}")


if __name__ == "__main__":
    main()
