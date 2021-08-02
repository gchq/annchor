import numpy as np
from annchor.datasets import load_digits
from annchor.distances import euclidean, levenshtein


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
