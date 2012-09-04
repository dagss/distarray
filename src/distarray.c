#include "distarray.h"

#include <stdlib.h>
#include <stdio.h>

distarray_grid_t *distarray_create_grid(MPI_Comm comm, int ndim, size_t *shape, int *ranks) {
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
  grid->shape = malloc(sizeof(size_t) * ndim);
  for (int idim = 0; idim < ndim; ++idim) {
    grid->shape[idim] = shape[idim];
  }
  grid->ranks = malloc(sizeof(int) * n);
  for (int i = 0; i < n; ++i) {
    grid->ranks[i] = ranks[i];
  }
  return grid;
}

void distarray_destroy_grid(distarray_grid_t *grid) {
  free(grid->shape);
  free(grid->ranks);
  free(grid);
}



