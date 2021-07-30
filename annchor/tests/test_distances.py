import numpy as np
from annchor.datasets import load_digits
from annchor.distances import euclidean, levenshtein, cosine, wasserstein


def test_levenshtein():

    # Check Levenshtein is as expected
    assert levenshtein("cat", "cart") == 1  # insertion
    assert levenshtein("cat", "cap") == 1  # substitution
    assert levenshtein("cat", "at") == 1  # deletion
    assert levenshtein("123456789", "92346781") == 3


def test_euclidean():

    x = np.random.random(100)
    y = np.random.random(100)
    assert np.isclose(euclidean(x, y), np.linalg.norm(x - y))


def test_cosine():
    x = np.random.random(100)
    y = np.random.random(100)
    assert np.isclose(
        cosine(x, y),
        1 - np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y)),
    )


def test_wasserstein():

    X = load_digits()["X"]
    assert np.isclose(wasserstein(X[0], X[645]), 0.7846513300484387)
    assert np.isclose(wasserstein(X[2], X[57]), 0.19850089919589903)
    assert np.isclose(wasserstein(X[3], X[673]), 1.6170254196506266)
    assert np.isclose(wasserstein(X[101], X[101]), 0.0)
