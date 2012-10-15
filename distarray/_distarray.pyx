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
    

cdef class Grid:
    cdef distarray_grid_t *grid

    def __init__(self, mpi4py.MPI.Comm comm, ranks):
        cdef int[:, ::1] _ranks = np.ascontiguousarray(ranks, dtype=np.intc)
        cdef int[::1] shape = np.ascontiguousarray(ranks.shape, dtype=np.intc)
        self.grid = distarray_create_grid(comm.ob_mpi, _ranks.ndim, &shape[0], &_ranks[0,0])
        if self.grid == NULL:
            raise ValueError("distarray_create_grid: Bad input")

    def __dealloc__(self):
        if self.grid != NULL:
            distarray_destroy_grid(self.grid)

    def get_rank(self, index):
        assert len(index) == 2 and isinstance(index, tuple)
        i, j = index
        return self.grid.ranks[i * self.grid.shape[1] + j]

    def find_rank(self, rank):
        cdef int[::1] coords = np.empty(self.grid.ndim, dtype=np.intc)
        distarray_grid_rank_to_coords(self.grid, rank, &coords[0])
        return tuple(coords)

if sizeof(cnp.npy_intp) != sizeof(size_t):
    raise NotImplementedError('sizeof(cnp.npy_intp) != sizeof(size_t)')
