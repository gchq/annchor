import sys
import numpy as np
import numba
import networkx as nx
from annchor.datasets import (
    load_digits,
    load_strings,
    load_digits_large,
    load_graph_sp,
)
from annchor.distances import levenshtein
from pynndescent.distances import kantorovich


def test_digits():
    data = load_digits()
    X = data["X"]
    y = data["y"]
    ng = data["neighbor_graph"]
    M = data["cost_matrix"]

    @numba.njit()
    def wasserstein(x, y):
        return kantorovich(x, y, cost=M)

    assert X.shape == (1797, 64)
    assert y.shape == (1797,)
    assert ng.shape == (2, 1797, 100)
    i = 10
    j = int(ng[0][i, 15])
    d = ng[1][i, 15]

    assert np.all(
        X[10]
        == np.array(
            [
                0.0,
                0.0,
                1.0,
                9.0,
                15.0,
                11.0,
                0.0,
                0.0,
                0.0,
                0.0,
                11.0,
                16.0,
                8.0,
                14.0,
                6.0,
                0.0,
                0.0,
                2.0,
                16.0,
                10.0,
                0.0,
                9.0,
                9.0,
                0.0,
                0.0,
                1.0,
                16.0,
                4.0,
                0.0,
                8.0,
                8.0,
                0.0,
                0.0,
                4.0,
                16.0,
                4.0,
                0.0,
                8.0,
                8.0,
                0.0,
                0.0,
                1.0,
                16.0,
                5.0,
                1.0,
                11.0,
                3.0,
                0.0,
                0.0,
                0.0,
                12.0,
                12.0,
                10.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                10.0,
                13.0,
                3.0,
                0.0,
                0.0,
            ]
        )
    )
    assert y[10] == 0
    assert np.all(np.isclose((i, j, d), (10, 676, 0.305587260000565)))
    assert np.isclose(wasserstein(X[i], X[j]), 0.305587260000565)


def test_digits_large():
    data = load_digits_large()
    X = data["X"]
    y = data["y"]
    ng = data["neighbor_graph"]
    M = data["cost_matrix"]

    @numba.njit()
    def wasserstein(x, y):
        return kantorovich(x, y, cost=M)

    assert X.shape == (5620, 64)
    assert y.shape == (5620,)
    assert ng.shape == (2, 5620, 100)
    i = 10
    j = int(ng[0][i, 15])
    d = ng[1][i, 15]

    assert np.all(
        X[10]
        == np.array(
            [
                0,
                0,
                6,
                14,
                14,
                16,
                16,
                8,
                0,
                0,
                7,
                11,
                8,
                10,
                15,
                3,
                0,
                0,
                0,
                0,
                4,
                15,
                10,
                0,
                0,
                1,
                15,
                16,
                16,
                16,
                14,
                0,
                0,
                3,
                11,
                13,
                13,
                0,
                0,
                0,
                0,
                0,
                0,
                15,
                5,
                0,
                0,
                0,
                0,
                0,
                7,
                13,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                12,
                0,
                0,
                0,
                0,
            ]
        )
    )
    assert y[10] == 7
    assert np.all(np.isclose((i, j, d), (10, 2939, 0.3401929036028718)))
    assert np.isclose(wasserstein(X[i], X[j]), 0.3401929036028718)


def test_load_strings():

    data = load_strings()
    X = data["X"]
    y = data["y"]
    ng = data["neighbor_graph"]

    assert X.shape == (1600,)
    assert y.shape == (1600,)
    assert ng.shape == (2, 1600, 100)
    i = 10
    j = int(ng[0][i, 15])
    d = ng[1][i, 15]

    assert X[10] == (
        "uofsjurgdrwshktxprvojrluttjiakqesuhdlkymvrjl"
        + "pvvhdiwdaxlwmsrqufzhnwilfphbhfsmeynyljkfxlud"
        + "bgkchhxojnwdrgiyytusxffdjbcyxtujppbtqivjealg"
        + "gvrkzfefodkfdddxjfwxalhtccozczukktoczzlyoexj"
        + "qycxswaiyxyvsflxxbitiwsinouhkxqlsgtuctflcayg"
        + "bcspwwpuefilrogmkcoyooslbmkpfzcenhpkqubkllif"
        + "pdwxgyyiiktppomptkadannpziipdocblolznfffjjwo"
        + "wpbsdumxmovbiuvrkfrotcksfmwiczpqulqaiuatjjvc"
        + "dwwawtrpucozosyqgzsidpmcedbtzabjdoqbpwcvusnw"
        + "qahjrouwhuyyzdulftasridctvedcsaqvvqwfraqwdpd"
        + "ixfillkdxummofoqzcohmgkjzonhulxeezwnwuzyncfm"
        + "enyeyhawhoqzgkwmu"
    )
    assert y[10] == 0
    assert (i, j, d) == (10, 165, 299)
    assert levenshtein(X[i], X[j]) == 299


def test_load_graph_sp():

    if float(sys.version[:3]) < 3.8:
        print('Skipping test_load_graph_sp (requires Python>=3.8)')
        return

    data = load_graph_sp()
    X = data["X"]
    y = data["y"]
    G = data["G"]
    ng = data["neighbor_graph"]

    assert X.shape == (800,)
    assert y.shape == (800,)
    assert ng.shape == (2, 800, 100)
    assert isinstance(G, nx.classes.graph.Graph)

    def sp_dist(i, j):
        return nx.dijkstra_path_length(G, i, j, weight="w")

    i = 10
    j = int(ng[0][i, 15])
    d = ng[1][i, 15]

    assert np.all(np.isclose((i, j, d), (10, 4, 0.3383337208609146)))
    assert np.isclose(sp_dist(X[i], X[j]), 0.3383337208609146)
