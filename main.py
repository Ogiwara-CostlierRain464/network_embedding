from typing import List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from helper import count
from sample import *
from scipy.spatial import distance

# number of nodes.
N = 6
# embedding dimensionality.
M = 2
# number of classes.
C = 2
# number of labels
L = 3
# τ
TAU = 5
# λ
LAMBDA = 0.5
# μ
MU = 0.01
# ρ
RHO = 0.01


def CF(x: np.ndarray, y: np.ndarray):
    return np.where(x == 0, y, x)


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
        woi = sum_mc - sum_mc # 0
        wo.append(woi)
    return np.column_stack(wo)


def WO2(W: np.ndarray, T: List[int], B: np.ndarray):
    assert W.shape == (M, C)
    assert B.shape == (M, N)
    assert len(T) == L

    # ここでのw_ciは、実際に得られたBベクトルを元にciを決定し、あとはそれを使ってw列ベクトルを選ぶ。

    wo = []
    sum_wc = W.sum(axis=1)
    for c in range(0, N):
        column = B[:, c]
        class_ = np.argmax(W.T @ column)
        wo.append(sum_wc - (W[:, class_] * C))
    return np.column_stack(wo)


def WO3(W: np.ndarray, T: List[int], B: np.ndarray):
    assert W.shape == (M, C)
    assert len(T) == L

    sum_mc = W.sum(axis=1)
    wo = []
    for _i in range(0, L):
        ci = T[_i]
        woi = sum_mc - (W[:, ci] * C)
        wo.append(woi)
    for r in range(L, N):
        may_be_class = np.argmax(W.T @ B[:, r])
        woi = sum_mc - (W[:, may_be_class] * C)
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
    b_sum = np.zeros((M, 1))
    for i in range(0, L):
        b_sum = b_sum + B[:, i].reshape((M, 1))

    for c in range(0, C):
        sum_1 = np.zeros((M, 1))
        for i in range(0, L):
            if T[i] == c:
                sum_1 = sum_1 + B[:, i].reshape((M, 1))

        w_c = sgn(sum_1 * C - b_sum)
        w_columns.append(w_c)
    return np.column_stack(w_columns)


def equation_13_was(B: np.ndarray, T: List[int]):
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


def loss(B: np.ndarray, S: np.ndarray, W: np.ndarray, T: List[int]):
    WO_ = WO(W, T)
    return -0.5 * np.trace(B @ S @ B.T) \
           + LAMBDA * np.trace(WO_.T @ B) \
           + MU * 0.25 * np.trace(B @ B.T) \
           + RHO * 0.5 * np.trace(B @ one(N))


def discrete_network_embedding(A: np.ndarray, T: List[int]):
    S = (A + (A @ A)) / 2
    W = np.random.uniform(-1, 1, (M, C))
    B = np.random.uniform(-1, 1, (M, N))
    # W = np.random.rand(M, C) # これは正の値しか出ない
    # B = np.random.rand(M, N)

    before_W = W
    before_B = B

    for _ in count(1, 20):
        for _ in count(1, 9):
            B = equation_11(B, S, W, T)
            print("updating B" + str(loss(B, S, W, T)))
        W = equation_13(B, T)
        print("updating W" + str(loss(B, S, W, T)))

    # x = []
    # y = []
    # colors = []
    # fig, ax = plt.subplots()
    # for i in range(0, N):
    #     b_i = B[:, i]
    #     x.append(b_i[0])
    #     y.append(b_i[1])
    #     ax.annotate(str(i), (b_i[0], b_i[1]))
    #     if np.argmax(W.T @ b_i) == 0:
    #         colors.append("r")
    #     else:
    #         colors.append("b")
    # for i in range(len(x)):
    #     ax.scatter(x[i], y[i], c=colors[i])
    # plt.show()
    return before_B, before_W, B, W


if __name__ == "__main__":
    Karate = nx.to_numpy_array(nx.karate_club_graph())
    A = Karate / Karate.sum(axis=1)[:, None]
    T = [0, 0, 0, 0, 0,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1]
    N = 34
    L = 15
    C = 2
    M = 500
    beforeB, beforeW, B, W = discrete_network_embedding(A, T)
    correct = [0, 0, 0, 0, 0,
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               1, 0, 0, 1, 0,
               1, 0, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1]
    answer = []
    for n in range(0, N):
        answer.append(np.argmax(W.T @ B[:, n]))

    S = (A + (A @ A)) / 2
    loss_before = loss(beforeB, S, beforeW, T)
    loss_after = loss(B, S, W, T)
    # assert loss_before > loss_after
    print("LOSS gain:" + str(loss_after - loss_before))
    print("H-distance:" + str(distance.hamming(answer, correct)))
    answer_top = answer[:L]
    print("H-distance of given:" + str(distance.hamming(answer_top, T)))



# Aをどうするかはわかった、問題は教師データの使い方