#include "distarray.h"

#include <stdlib.h>
#include <stdio.h>

distarray_grid_t *distarray_create_grid(MPI_Comm comm, int ndim, int *shape, int *ranks) {
  distarray_grid_t *grid = malloc(sizeof(distarray_grid_t));
  size_t n = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    n *= shape[idim];
  }
  int comm_size;
  if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS) {
    return NULL;
  }
  if (n != comm_size) {
    return NULL;
  }
  grid->comm = comm;
  grid->ndim = ndim;
  grid->shape = malloc(sizeof(int) * ndim);
  for (int idim = 0; idim < ndim; ++idim) {
    grid->shape[idim] = shape[idim];
  }
  grid->ranks = malloc(sizeof(int) * n);
  for (int i = 0; i < n; ++i) {
    grid->ranks[i] = ranks[i];
  }
  grid->strides = malloc(sizeof(int) * ndim);
  grid->strides[ndim - 1] = 1;
  for (int idim = ndim - 2; idim >= 0; --idim) {
    grid->strides[idim] = grid->strides[idim + 1] * grid->shape[idim + 1];
  }

  return grid;
}

void distarray_destroy_grid(distarray_grid_t *grid) {
  free(grid->shape);
  free(grid->ranks);
  free(grid->strides);
  free(grid);
}

int distarray_grid_rank_to_coords(distarray_grid_t *grid, int rank, int *coords) {
    int comm_size;
    if (!mpi_check(MPI_Comm_size(grid->comm, &comm_size))) return -1;

    /* find `iflat`, the index of `rank` in `grid->ranks` */
    int iflat;
       
    for (iflat = 0; iflat < comm_size; ++iflat) {
        if (grid->ranks[iflat] == rank) break;
    }
    if (iflat == comm_size) return -1;

    /* return coordinate `iflat` corresponds to in ND grid */
    for (int idim = 0; idim < grid->ndim; ++idim) {
        coords[idim] = iflat / grid->strides[idim];
        iflat %= grid->strides[idim];
    }
    return 0;
}
