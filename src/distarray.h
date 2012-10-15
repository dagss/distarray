#ifndef _e8573ca9_e599_4f27_a05f_a46e6864ab8f
#define _e8573ca9_e599_4f27_a05f_a46e6864ab8f

#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct {
    MPI_Comm comm;
    int ndim;
    int *shape;
    int *strides;
    int *ranks;
} distarray_grid_t;

distarray_grid_t *distarray_create_grid(MPI_Comm comm, int ndim, int *shape, int *ranks);
void distarray_destroy_grid(distarray_grid_t *grid);
int distarray_grid_rank_to_coords(distarray_grid_t *grid, int rank, int *coords);


#endif