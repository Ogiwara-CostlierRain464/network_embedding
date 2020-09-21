from typing import List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from helper import count
from sample import *

# number of nodes.
N = 12
# embedding dimensionality.
M = 2
# number of classes.
C = 3
# number of labels
L = 9
# τ
TAU = 5
# λ
LAMBDA = 0.5
# μ
MU = 0.01
# ρ
RHO = 0.01


def CF(x: np.ndarray, y: np.ndarray):
    if not np.any(x):  # if some of element is not 0
        return x
    else:
        return y


def sgn(x: np.ndarray):
    x_sign = np.sign(x)  # convert to {+1, 0} matrix.
    return np.where(x_sign == 0, -1, x_sign)  # change all 0 to -1.


def WO(W: np.ndarray, T: List[int]):
    assert W.shape == (M, C)
    assert len(T) == L

    w_mean = W.mean(axis=1)
    sum_mc = W.sum(axis=1)
    wo = []
    for _i in range(0, L):
        ci = T[_i]
        woi = sum_mc - (W[:, ci] * C)
        wo.append(woi)
    for _ in count(L + 1, N):
        woi = sum_mc - (w_mean * C)
        wo.append(woi)
    return np.column_stack(wo)


def one(dim: int):
    return np.ones(dim).reshape(N, 1)


def equation_11(B: np.ndarray, S: np.ndarray, W: np.ndarray, T: List[int]):
    assert B.shape == (M, N)
    assert S.shape == (N, N)
    assert W.shape == (M, C)
    dLB = - B @ S + WO(W, T) * LAMBDA + (B @ B.T @ B) * MU + (B @ one(N) @ one(N).T) * RHO
    return sgn(CF(B * TAU - dLB, B))


def equation_13(B: np.ndarray, T: List[int]):
    w_columns = []
    sum_2 = B.sum(axis=1).reshape((M, 1))
    for c in range(0, C):
        sum_1 = np.zeros((M, 1))
        for i in range(0, L):
            if T[i] == c:
                sum_1 = sum_1 + B[:, i].reshape((M, 1))

        w_c = sgn(sum_1 * C - sum_2)
        w_columns.append(w_c)
    return np.column_stack(w_columns)


def discrete_network_embedding(A: np.ndarray, T: List[int]):
    S = (A + (A @ A)) / 2
    W = np.random.rand(M, C)
    B = np.random.uniform(-1, 1, (M, N))
    B[:, 0] = [-1, -1]
    B[:, 1] = [+1, +1]
    B[:, 2] = [-1, -1]

    for _ in count(1, 100):
        for _ in count(1, 100):
            B = equation_11(B, S, W, T)
        W = equation_13(B, T)

    plt.scatter(B[0], B[1])
    plt.show()
    return B, W


if __name__ == "__main__":
    G = sample4()
    A = nx.to_numpy_array(G)
    T = [0, 1, 0]
    N = 6
    L = 3
    C = 2
    B, W = discrete_network_embedding(A, T)
