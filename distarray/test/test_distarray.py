from .common import *
from nose.tools import assert_raises

import distarray

import numpy as np

@mpi(6)
def test_grid(comm):
    ranks = np.arange(6, dtype=np.intc).reshape(2, 3)
    g = distarray.Grid(comm, ranks)
    ranks[...] = 0
    for i in range(2):
        for j in range(3):
            assert 3 * i + j == g.get_rank((i, j))
            assert g.find_rank(3 * i + j) == (i, j)

@mpi(6)
def test_grid_bad_size(comm):
    ranks = np.arange(8, dtype=np.intc).reshape(2, 4)
    with assert_raises(ValueError):
        distarray.Grid(comm, ranks)
