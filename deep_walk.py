from typing import List

import matplotlib.pyplot as plt
import random
import networkx as nx
from gensim.models import Word2Vec as word2vec
from sample import *


def make_random_walks(G: nx.Graph, num_walk, length_of_walk) -> List[List[str]]:
    # ランダムウォークで歩いたノードを入れるlistを生成
    paths = list()
    # ランダムウォークを擬似的に行う
    for i in range(num_walk):
        node_list = list(G.nodes())
        for node in node_list:
            now_node = node
            # 到達したノードを追加する用のリストを用意する
            path = list()
            path.append(str(now_node))
            for j in range(length_of_walk):
                # 次に到達するノードを選択する
                next_node = random.choice(list(G.neighbors(now_node)))
                # リストに到達したノードをリストに追加する
                path.append(str(next_node))
                # 今いるノードを「現在地」とする
                now_node = next_node
            # ランダムウォークしたノードをリストに追加
            paths.append(path)
        # 訪れたノード群を返す
        return paths


G = sample4()
walking = make_random_walks(G, num_walk=512, length_of_walk=1000)
model = word2vec(walking, min_count=1, size=2, window=10, workers=1)

x = []
y = []
node_list = []
colors = []
fig, ax = plt.subplots()
for node in G.nodes:
    vector = model.wv[str(node)]
    x.append(vector[0])
    y.append(vector[1])
    ax.annotate(str(node), (vector[0], vector[1]))
    if 0 <= node <= 2:
        colors.append("r")
    else:
        colors.append("b")
for i in range(len(x)):
    ax.scatter(x[i], y[i], c=colors[i])
plt.show()


# Node内の情報はどこで使ってる？単にContextの情報を使っているようにしか見えないけど。