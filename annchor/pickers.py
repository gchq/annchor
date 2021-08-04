from abc import ABC, abstractmethod

import numpy as np


from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import os
from annchor.utils import *
from numba import njit, prange, types
from numba.typed import Dict

from tqdm.auto import tqdm as tq

CPU_COUNT = os.cpu_count()


class MaxMinAnchorPicker:
    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = np.zeros(na).astype(int)
        ix = np.random.randint(nx)

        if ann.verbose:

            def v(f):
                return tq(f)

        else:

            def v(f):
                return f

        for i in v(range(na)):
            A[i] = ix
            IJs = np.array([[ix, j] for j in range(ann.nx)])
            D[i] = ann.get_exact_ijs(ann.f, ann.X, IJs)
            if i == 0:
                ix = np.argmax(np_min(D, 0))
            else:
                ix = np.argmax(np_min(D[1:], 0))

        return A, D.T, na * nx


class ExternalAnchorPicker:
    def __init__(self, A):
        self.A = A
        self.is_anchor_safe = False

    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        if ann.verbose:

            def v(f):
                return tq(f)

        else:

            def v(f):
                return f

        for i in v(range(na)):
            D[i] = np.array([ann.f(x, self.A[i]) for x in ann.X])

        return np.array([]), D.T, na * nx


class SelectedAnchorPicker:
    def __init__(self, A):
        self.A = A

    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = self.A

        IJ = np.array([[i, j] for i in A for j in range(nx)])
        D = ann.get_exact_ijs(ann.f, ann.X, IJ)
        D = D.reshape(na, nx)

        return A, D.T, na * nx


class RandomAnchorPicker:
    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = np.random.choice(np.arange(nx), na, replace=False)

        IJ = np.array([[i, j] for i in A for j in range(nx)])
        D = ann.get_exact_ijs(ann.f, ann.X, IJ)
        D = D.reshape(na, nx)

        return A, D.T, na * nx
