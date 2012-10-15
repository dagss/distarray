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

@mpi(2)
def test_chunked_axis(comm):
    grid = distarray.Grid(comm, (2,))
    dist = distarray.Distribution(grid, [
        distarray.ChunkedAxis([0, 200, 400])
        ], [1])
    local = np.arange(20)[None, :].copy()

    # local_to_global
    global_ = dist.local_to_global(local)
    start = 200 * comm.Get_rank()
    assert np.all(global_[0, :] == np.arange(start, start + 20))

    # global_to_rank
    global_ = np.arange(195, 205)[None, :].copy()
    assert np.all(dist.global_to_rank_coords(global_) ==
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
