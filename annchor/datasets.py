import os
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))


def load_digits():

    """load_digits

    Loads the UCI OCR digits data test set (1797 8x8 images).

    https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science.

    Parameters
    ----------

    Returns
    -------
    digits_data: dict, keys=['X', 'y', 'neighbor_graph']
        The string data set.

        digits_data['X'] is an np.array ints, shape=(1797,64)

        digits_data['y'] is an np.array of ints (labels), shape=(1797,)

        digits_data['neighbor_graph'] is a tuple containing the 100-NN graph
        data

        digits_data['neighbor_graph'][0] are the 100-NN indices,
        i.e. digits_data['neighbor_graph'][0][i][j] is the jth nearest index to
        index i.

        digits_data['neighbor_graph'][1] are the 100-NN distances,
        i.e. digits_data['neighbor_graph'][0][i][j] is the jth smallest
        distance to index i.
    """

    file = os.path.join(package_directory, "data", "digits_data.npz")
    digits_data = np.load(file)

    return digits_data


def load_digits_large():

    """load_digits_large

    Loads the full (train+test) UCI OCR digits data set (5620 8x8 images).

    https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science.

    Parameters
    ----------

    Returns
    -------
    digits_data: dict, keys=['X', 'y', 'neighbor_graph']
        The string data set.

        digits_data['X'] is an np.array ints, shape=(5620,64)

        digits_data['y'] is an np.array of ints (labels), shape=(5620,)

        digits_data['neighbor_graph'] is a tuple containing the 100-NN graph
        data

        digits_data['neighbor_graph'][0] are the 100-NN indices,
        i.e. digits_data['neighbor_graph'][0][i][j] is the jth nearest index to
        index i.

        digits_data['neighbor_graph'][1] are the 100-NN distances,
        i.e. digits_data['neighbor_graph'][0][i][j] is the jth smallest
        distance to index i.
    """

    file = os.path.join(package_directory, "data", "digits_data_large.npz")
    digits_data = np.load(file)

    return digits_data


def load_strings():

    """load_strings

    Loads the string data set (1600 strings, length ~500). There are
    8 clusters of two types: clouds (labels 0-4) and filaments (labels 5-7).

    Parameters
    ----------

    Returns
    -------
    strings_data: dict, keys=['X', 'y', 'neighbor_graph']
        The string data set.

        strings_data['X'] is an np.array of strings, shape=(1600,)

        strings_data['y'] is an np.array of ints (labels), shape=(1600,)

        strings_data['neighbor_graph'] is a tuple containing the 100-NN graph
        data

        strings_data['neighbor_graph'][0] are the 100-NN indices,
        i.e. strings_data['neighbor_graph'][0][i][j] is the jth nearest index
        to index i.

        strings_data['neighbor_graph'][1] are the 100-NN distances,
        i.e. strings_data['neighbor_graph'][0][i][j] is the jth smallest
        distance to index i.
    """

    file = os.path.join(package_directory, "data", "strings_data.npz")
    string_data = np.load(file)

    return string_data


def load_graph_sp():

    """load_graph_sp

    Loads the graph shortest path data set (800 vertices from a weighted
    graph).
    There are 10 clusters of vertices. The intra-cluster distances are shorter
    on average than the inter-cluster distances.

    Parameters
    ----------

    Returns
    -------
    graph_sp_data: dict, keys=['X', 'y', 'neighbor_graph']
        The string data set.

        graph_sp_data['X'] is an np.array of vertex indices, shape=(800,)

        graph_sp_data['y'] is an np.array of ints (labels), shape=(800,)

        graph_sp_data['neighbor_graph'] is a tuple containing the 100-NN graph
        data

        graph_sp_data['neighbor_graph'][0] are the 100-NN indices,
        i.e. graph_sp_data['neighbor_graph'][0][i][j] is the jth nearest index
        to index i.

        graph_sp_data['neighbor_graph'][1] are the 100-NN distances,
        i.e. graph_sp_data['neighbor_graph'][0][i][j] is the jth smallest
        distance to index i.
    """

    file = os.path.join(package_directory, "data", "graph_sp_data.npz")
    graph_sp_data = np.load(file)
    try:
        import networkx as nkx

        data = np.load(os.path.join(package_directory, "data", "graph.npz"))
        edge_list = [
            "%d %d %s" % (i, j, w)
            for (i, j), w in zip(data["edges"], data["weights"])
        ]
        G = nkx.readwrite.edgelist.parse_edgelist(
            edge_list, nodetype=int, data=(("w", float),)
        )

    except ImportError as E:
        raise Exception("Error: load_graph_sp requires networkx.")

    return {
        "X": graph_sp_data["X"],
        "y": graph_sp_data["y"],
        "neighbor_graph": graph_sp_data["neighbor_graph"],
        "G": G,
    }
