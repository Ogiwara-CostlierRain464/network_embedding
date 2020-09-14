import numpy as np
import networkx as nx


def C(x, y):
    if x != 0:
        return x
    else:
        return y


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


def W0(W: np.ndarray):
    return W


def equation_11(B: np.ndarray, S: np.ndarray, W: np.ndarray):
    pass


def equation_13(B: np.ndarray):
    pass


def discrete_network_embedding(A: np.ndarray):
    n = 12
    m = 3
    C = 2
    S = (A + (A @ A))/2
    W = np.random.rand(m, C)
    B = np.random.rand(m, n)
    for _ in range(1, 100+1):
        for _ in range(1, 10):
            pass


if __name__ == "__main__":
    G = nx.Graph()
    for i in range(0, 11 + 1):
        G.add_node(i)
    G.add_edges_from([
        (0, 1, {"weight": 3}),
        (0, 2, {"weight": 3}),
        (1, 2, {"weight": 3})
    ])
    G.add_edges_from([
        (0, 4, {"weight": 4}),
        (0, 5, {"weight": 7})
    ])
    G.add_edges_from([
        (1, 6, {"weight": 5}),
        (1, 7, {"weight": 5}),
        (1, 8, {"weight": 4}),
        (8, 7, {"weight": 10}),
        (1, 9, {"weight": 6}),
    ])
    G.add_edges_from([
        (2, 10, {"weight": 5}),
        (2, 11, {"weight": 5}),
        (2, 3, {"weight": 5}),
        (11, 10, {"weight": 10}),
    ])
    A = nx.to_numpy_array(G)



