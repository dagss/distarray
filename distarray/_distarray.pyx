from libc.stddef cimport size_t, ptrdiff_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cimport mpi4py.MPI

cimport numpy as cnp
import numpy as np

cdef extern from "mpi.h":
    ctypedef int MPI_Comm

cdef extern from "distarray.h":
    
    ctypedef struct distarray_grid_t:
        MPI_Comm comm
        int ndim
        int *shape
        int *strides
        int *ranks
        
    distarray_grid_t *distarray_create_grid(MPI_Comm comm, int ndim, int *shape, int *ranks)
    int distarray_grid_rank_to_coords(distarray_grid_t *grid, int rank, int *coords)
    void distarray_destroy_grid(distarray_grid_t *grid)

    ctypedef struct distarray_distribution_t:
        distarray_grid_t *grid

    distarray_distribution_t *distarray_create_distribution(distarray_grid_t *grid)
    void distarray_destroy_distribution(distarray_distribution_t *dist)

    int distarray_add_chunked_axis(distarray_distribution_t *dist, ptrdiff_t len_,
                                   ptrdiff_t stride, ptrdiff_t *offsets)

    void distarray_local_to_global(distarray_distribution_t *dist, ptrdiff_t n,
                                   ptrdiff_t * local, ptrdiff_t * global_)
    void distarray_local_shape(distarray_distribution_t *dist, int rank, ptrdiff_t *shape)

    void distarray_global_to_rank_coords(distarray_distribution_t *dist, ptrdiff_t n,
                                         ptrdiff_t * global_, int * rank_coords)

    ctypedef struct distarray_plan_t:
        pass
    
    distarray_plan_t *distarray_plan_redistribution(distarray_distribution_t *from_dist,
                                                    distarray_distribution_t *to_dist)
    void distarray_destroy_plan(distarray_plan_t *plan)
    int distarray_execute(distarray_plan_t *plan, void *from_buf, void *to_buf)
    void distarray_send_counts(distarray_distribution_t *from_dist,
                               distarray_distribution_t *to_dist,
                               int from_rank, int comm_size, int *counts)

    

if sizeof(size_t) == 4:
    size_t_dtype = np.uint32
elif sizeof(size_t) == 8:
    size_t_dtype = np.uint64
else:
    assert False

if sizeof(ptrdiff_t) == 4:
    ptrdiff_dtype = np.int32
elif sizeof(ptrdiff_t) == 8:
    ptrdiff_dtype = np.int64
else:
    assert False

cdef class Grid:
    cdef distarray_grid_t *grid
    cdef mpi4py.MPI.Comm comm

    def __init__(self, mpi4py.MPI.Comm comm, ranks):
        cdef cnp.ndarray _ranks
        if isinstance(ranks, tuple):
            _ranks = np.arange(comm.Get_size(), dtype=np.intc).reshape(ranks)
        else:
            _ranks = np.ascontiguousarray(ranks, dtype=np.intc)
        cdef int[::1] shape = np.ascontiguousarray((<object>_ranks).shape, dtype=np.intc)
        self.grid = distarray_create_grid(comm.ob_mpi, _ranks.ndim, &shape[0],
                                          <int*>_ranks.data)
        if self.grid == NULL:
            raise ValueError("distarray_create_grid: Bad input")
        self.comm = comm

    def __dealloc__(self):
        if self.grid != NULL:
            distarray_destroy_grid(self.grid)
            self.grid = NULL

    def get_rank(self, index):
        assert len(index) == 2 and isinstance(index, tuple)
        i, j = index
        return self.grid.ranks[i * self.grid.shape[1] + j]

    def find_rank(self, rank):
        cdef int[::1] coords = np.empty(self.grid.ndim, dtype=np.intc)
        distarray_grid_rank_to_coords(self.grid, rank, &coords[0])
        return tuple(coords)

cdef class ChunkedAxis:
    cdef ptrdiff_t[::1] offsets
    
    def __init__(self, offsets):
        self.offsets = np.ascontiguousarray(offsets, dtype=ptrdiff_dtype)

    def associate(self, Distribution dist, length, stride):
        distarray_add_chunked_axis(dist.dist, length, stride, &self.offsets[0]);


cdef class Distribution:
    cdef distarray_distribution_t *dist
    cdef object axes, shape
    cdef Grid grid
    
    def __cinit__(self, Grid grid, axes, shape, strides):
        self.grid = grid
        self.dist = distarray_create_distribution(grid.grid)
        if self.dist == NULL:
            raise RuntimeError()
        self.axes = list(axes)
        self.shape = tuple(shape)
        for axis, length, stride in zip(self.axes, shape, strides):
            axis.associate(self, length, stride)            

    def __dealloc__(self):
        if self.dist != NULL:
            distarray_destroy_distribution(self.dist)
            self.dist = NULL

    def local_shape(self):
        my_rank = self.grid.comm.Get_rank()
        cdef ptrdiff_t[:] shape = np.empty(self.dist.grid.ndim, dtype=ptrdiff_dtype)
        distarray_local_shape(self.dist, my_rank, &shape[0])
        return tuple(shape)

    def local_to_global(self, ptrdiff_t[:, ::1] local_indices):
        if local_indices.shape[0] != self.dist.grid.ndim:
            raise ValueError()
        cdef ptrdiff_t[:, ::1] global_indices = np.empty(
            (local_indices.shape[0], local_indices.shape[1]),
            ptrdiff_dtype)

        distarray_local_to_global(self.dist, local_indices.shape[1],
                                  &local_indices[0,0], &global_indices[0,0])

        return np.asarray(global_indices)

    def global_to_rank_coords(self, ptrdiff_t[:, ::1] global_indices):
        if global_indices.shape[0] != self.dist.grid.ndim:
            raise ValueError()
        cdef int[:, ::1] ranks = np.empty(
            (global_indices.shape[0], global_indices.shape[1]),
            np.intc)

        distarray_global_to_rank_coords(self.dist, global_indices.shape[1],
                                        &global_indices[0,0], &ranks[0,0])

        return np.asarray(ranks)
        
cdef class RedistributionPlan:
    cdef distarray_plan_t *plan
    cdef Distribution from_dist, to_dist
    cdef object dtype

    def __cinit__(self, Distribution from_dist, Distribution to_dist, dtype):
        self.plan = distarray_plan_redistribution(from_dist.dist, to_dist.dist)
        self.from_dist = from_dist
        self.to_dist = to_dist
        self.dtype = dtype

    def __dealloc__(self):
        if self.plan != NULL:
            distarray_destroy_plan(self.plan)
            self.plan = NULL

    def execute(self, cnp.ndarray from_buf, cnp.ndarray to_buf):
        if (<object>from_buf).shape != self.from_dist.local_shape() or from_buf.dtype != self.dtype:
            raise ValueError()
        if (<object>to_buf).shape != self.to_dist.local_shape() or to_buf.dtype != self.dtype:
            raise ValueError()
        distarray_execute(self.plan, <void*>from_buf.data, <void*>to_buf.data)

def compute_send_counts(int rank, Distribution from_dist, Distribution to_dist):
    cdef int n = from_dist.grid.comm.Get_size()
    cdef int[:] counts = np.zeros(n, dtype=np.intc)
    # todo: assertions
    distarray_send_counts(from_dist.dist, to_dist.dist, rank, n,
                          &counts[0])
    return np.asarray(counts)
