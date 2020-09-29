import networkx as nx
import urllib.request as urllib
import io
import zipfile

import matplotlib.pyplot as plt



def sample1():
    G = nx.Graph()
    for _i in range(0, 11 + 1):
        G.add_node(_i)
    G.add_edges_from([
        (0, 1, {"weight": 3}),
        (0, 2, {"weight": 3}),
        (1, 2, {"weight": 3})
    ])
    G.add_edges_from([
        (0, 4, {"weight": 4}),
        (0, 5, {"weight": 7}),
        (4, 5, {"weight": 10}),
    ])
    G.add_edges_from([
        (1, 6, {"weight": 5}),
        (1, 7, {"weight": 5}),
        (1, 8, {"weight": 4}),
        (8, 7, {"weight": 10}),
        (1, 9, {"weight": 6}),
        (6, 7, {"weight": 10}),
        (8, 9, {"weight": 10}),
        (7, 9, {"weight": 10}),
    ])
    G.add_edges_from([
        (2, 10, {"weight": 5}),
        (2, 11, {"weight": 5}),
        (2, 3, {"weight": 5}),
        (11, 10, {"weight": 10}),
        (11, 3, {"weight": 11}),
    ])
    return G


def sample2():
    G = nx.Graph()
    for i in range(0, 11 + 1):
        G.add_node(i)
    G.add_edges_from([
        (0, 1, {"weight": 10}),
        (0, 2, {"weight": 11}),
        (1, 2, {"weight": 9})
    ])
    G.add_edges_from([
        (3, 4, {"weight": 7}),
        (3, 5, {"weight": 5}),
        (4, 5, {"weight": 10}),
        (4, 6, {"weight": 9}),
        (5, 6, {"weight": 8}),
    ])
    G.add_edges_from([
        (7, 8, {"weight": 6}),
        (7, 9, {"weight": 8}),
        (7, 10, {"weight": 10}),
        (7, 11, {"weight": 10}),
        (8, 9, {"weight": 9}),
        (8, 10, {"weight": 7}),
        (9, 11, {"weight": 11}),
        (10, 11, {"weight": 5}),
    ])
    return G


def sample3():
    G = nx.Graph()
    for i in range(0, 8 + 1):
        G.add_node(i)
    G.add_edges_from([
        (0, 3, {"weight": 1}),
        (0, 4, {"weight": 1}),
        (3, 4, {"weight": 1})
    ])
    G.add_edges_from([
        (1, 5, {"weight": 1}),
        (1, 6, {"weight": 1}),
        (5, 6, {"weight": 1}),
    ])
    G.add_edges_from([
        (2, 7, {"weight": 1}),
        (2, 8, {"weight": 1}),
        (7, 8, {"weight": 1}),
    ])
    return G


def sample4():
    G = nx.Graph()
    for i in range(0, 5 + 1):
        G.add_node(i)
    G.add_edges_from([
        (0, 2, {"weight": 1}),
        (0, 4, {"weight": 1}),
        (2, 4, {"weight": 1})
    ])
    G.add_edges_from([
        (1, 3, {"weight": 1}),
        (1, 5, {"weight": 1}),
        (3, 5, {"weight": 1}),
    ])
    return G


def sample5():
    G = nx.karate_club_graph()
    return G

def football():
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

    sock = urllib.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read("football.txt").decode()  # read info file
    gml = zf.read("football.gml").decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split("\n")[1:]
    G = nx.parse_gml(gml)  # parse gml data
    return G