from .common import *
from nose.tools import assert_raises

import distarray

import numpy as np

@mpi(6)
def test_grid(comm):
    ranks = np.arange(6, dtype=np.intc).reshape(2, 3)
    g = distarray.Grid(comm, ranks)
    ranks[...] = 0
    assert 5 == g.get_rank((1, 2))

@mpi(6)
def test_grid_bad_size(comm):
    ranks = np.arange(8, dtype=np.intc).reshape(2, 4)
    with assert_raises(ValueError):
        distarray.Grid(comm, ranks)
