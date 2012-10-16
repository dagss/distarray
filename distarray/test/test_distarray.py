from .common import *
from nose.tools import assert_raises

from distarray import Grid, Distribution, ChunkedAxis, RedistributionPlan
import distarray

import numpy as np

@mpi(6)
def test_grid(comm):
    ranks = np.arange(6, dtype=np.intc).reshape(2, 3)
    g = Grid(comm, ranks)
    ranks[...] = 0
    for i in range(2):
        for j in range(3):
            assert 3 * i + j == g.get_rank((i, j))
            assert g.find_rank(3 * i + j) == (i, j)

@mpi(6)
def test_grid_bad_size(comm):
    ranks = np.arange(8, dtype=np.intc).reshape(2, 4)
    with assert_raises(ValueError):
        Grid(comm, ranks)

@mpi(2)
def test_chunked_axis(comm):
    grid = Grid(comm, (2,))
    dist = Distribution(grid, [
        ChunkedAxis([0, 200, 400])
        ], shape=(400,), strides=(1,))
    local = np.arange(20)[None, :].copy()

    assert dist.local_shape() == (200,)

    # local_to_global
    global_ = dist.local_to_global(local)
    start = 200 * comm.Get_rank()
    assert np.all(global_[0, :] == np.arange(start, start + 20))

    # global_to_rank
    global_ = np.arange(195, 205)[None, :].copy()
    assert np.all(dist.global_to_rank_coords(global_) ==
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

@mpi(6)
def test_chunked_execute(comm):
    rank = comm.Get_rank()
    grid = Grid(comm, (2, 3))

    dist_a = Distribution(grid, [
        ChunkedAxis([0, 10, 40]),
        ChunkedAxis([0, 5, 10, 15])
        ], shape=(40, 15), strides=(15, 1))

    dist_b = Distribution(grid, [
        ChunkedAxis([0, 30, 40]),
        ChunkedAxis([0, 10, 13, 15])
        ], shape=(40, 15), strides=(15, 1))

    # Local shape
    meq_(comm, [(10, 5), (10, 5), (10, 5),
                (30, 5), (30, 5), (30, 5)], dist_a.local_shape())

    meq_(comm, [(30, 10), (30, 3), (30, 2),
                (10, 10), (10, 3), (10, 2)], dist_b.local_shape(), )


    # sendcnts; all results computed on all nodes
    sendcounts = np.zeros((6, 6), int)
    for i in range(6):
        sendcounts[i,:] = distarray.compute_send_counts(i, dist_a, dist_b)

    assert np.all(sendcounts == [
        [ 50,  0,  0,  0,  0,  0],
        [ 50,  0,  0,  0,  0,  0],
        [  0, 30, 20,  0,  0,  0],
        [100,  0,  0, 50,  0,  0],
        [100,  0,  0, 50,  0,  0],
        [ 0, 60, 40,  0, 30, 20]])

    # recvcnts; all results computed on all nodes
    recvcounts = np.zeros((6, 6), int)
    for i in range(6):
        recvcounts[i,:] = distarray.compute_send_counts(i, dist_b, dist_a)
    assert np.all(recvcounts == sendcounts.T)
    
    plan = RedistributionPlan(dist_a, dist_b, np.double)
    arr_a = np.ones(dist_a.local_shape()) * rank
    arr_b = np.empty(dist_b.local_shape()) * np.nan
#    mprint(comm, arr_a)
    #plan.execute(arr_a, arr_b)
    mprint(comm, arr_b)
    
## @mpi(6)
## def test_chunked_axis_2d(comm):
##     # 2D
##     grid = Grid(comm, (2, 3))
##     dist = Distribution(grid, [
##         ChunkedAxis([0, 100, 200])
##         ChunkedAxis([0, 100, 200, 300])
##         ], [300, 1])

    
