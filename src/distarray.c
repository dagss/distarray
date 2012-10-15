#include "distarray.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    if (MPI_Comm_size(grid->comm, &comm_size) != MPI_SUCCESS) return -1;

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


/*
distribution
*/
distarray_distribution_t *distarray_create_distribution(distarray_grid_t *grid) {
    int ndim = grid->ndim;
    distarray_distribution_t *dist = malloc(sizeof(distarray_distribution_t) + /* struct */
                                            sizeof(void*) * ndim + /* axes array */
                                            sizeof(ptrdiff_t) * ndim); /* strides array */
    dist->grid = grid;
    dist->axes_added = 0;

    char *head = (char*)dist + sizeof(distarray_distribution_t);
    dist->axes = (distarray_axis_t**)head;
    head += ndim * sizeof(void*);
    dist->strides = (ptrdiff_t*)head;

    memset(dist->axes, 0, sizeof(void*) * ndim);
    memset(dist->strides, 0, sizeof(ptrdiff_t) * ndim);

    return dist;
}

void distarray_destroy_distribution(distarray_distribution_t *dist) {
    for (int i = 0; i != dist->axes_added; ++i) {
        dist->axes[i]->destructor(dist->axes[i]);
    }
    free(dist);
}

int distarray_distribution_adopt_axis(distarray_distribution_t *dist, distarray_axis_t *axis,
                                      ptrdiff_t stride) {
    if (dist->axes_added == dist->grid->ndim) return -1;
    dist->axes[dist->axes_added] = axis;
    dist->strides[dist->axes_added] = stride;
    dist->axes_added++;
    return 0;
}

/*
axis
*/

static void _default_axis_destructor(distarray_axis_t *axis) {
    free(axis);
}

/*
 chunked axis
*/

typedef struct {
    distarray_axis_t base;
    ptrdiff_t *offsets;
    ptrdiff_t local_start, local_stop;
} distarray_chunked_axis_t;


static void _chunked_local_to_global(distarray_axis_t *restrict self,
                                     ptrdiff_t n,
                                     ptrdiff_t *restrict local_indices,
                                     ptrdiff_t *restrict global_indices) {
    ptrdiff_t local_start = ((distarray_chunked_axis_t *)self)->local_start;
    
    for (ptrdiff_t i = 0; i != n; ++i) {
        global_indices[i] = local_indices[i] + local_start;
    }
}

static void _chunked_global_to_rank_coords(distarray_axis_t *restrict self,
                                           ptrdiff_t n,
                                           ptrdiff_t *restrict global,
                                           int *restrict rank_coords) {
    ptrdiff_t *offsets = ((distarray_chunked_axis_t*)self)->offsets;
    int ichunk = 0;
    for (ptrdiff_t i = 0; i != n; ++i) {
        ptrdiff_t gidx = global[i];
        if (gidx < offsets[ichunk]) {
            while (gidx < offsets[--ichunk]) {}
        } else if (gidx >= offsets[ichunk + 1]) {
            while (gidx >= offsets[++ichunk + 1]) {}
        }
        rank_coords[i] = ichunk;
    }
}

int distarray_add_chunked_axis(distarray_distribution_t *dist, ptrdiff_t *offsets,
                               ptrdiff_t stride) {
    distarray_chunked_axis_t *axis = malloc(sizeof(distarray_chunked_axis_t));
    axis->offsets = offsets;
    axis->base.destructor = &_default_axis_destructor;
    axis->base.local_to_global = &_chunked_local_to_global;
    axis->base.global_to_rank_coords = &_chunked_global_to_rank_coords;

    int iaxis = dist->axes_added;
    
    int rank;
    if (MPI_Comm_rank(dist->grid->comm, &rank) != MPI_SUCCESS) return -1;
    int grid_coords[dist->grid->ndim];
    if (distarray_grid_rank_to_coords(dist->grid, rank, grid_coords) != 0) return -1;
    axis->local_start = offsets[grid_coords[iaxis]];
    axis->local_stop = offsets[grid_coords[iaxis] + 1];

    distarray_distribution_adopt_axis(dist, (void*)axis, stride);

    return 0;
}


/* 
   plan
 */

distarray_plan_t *distarray_plan_redistribution(distarray_distribution_t *from_dist,
                                                distarray_distribution_t *to_dist) {
    distarray_plan_t *plan = malloc(sizeof(distarray_plan_t));
    plan->from = from_dist;
    plan->to = to_dist;
    return plan;
}

void distarray_destroy_plan(distarray_plan_t *plan) {
    free(plan);
}

int distarray_execute(distarray_plan_t *plan, void *from_buf, void *to_buf) {

}

void distarray_local_to_global(distarray_distribution_t *dist, ptrdiff_t n,
                               ptrdiff_t *restrict local, ptrdiff_t *restrict global) {
    ptrdiff_t i;
    int ndim = dist->grid->ndim;

    for (int iax = 0; iax != ndim; ++iax) {
        distarray_axis_t *axis = dist->axes[iax];
        (*axis->local_to_global)(axis, n, local, global);
    }
}

void distarray_global_to_rank_coords(distarray_distribution_t *dist, ptrdiff_t n,
                                     ptrdiff_t *restrict global, int *restrict rank_coords) {
    ptrdiff_t i;
    int ndim = dist->grid->ndim;

    for (int iax = 0; iax != ndim; ++iax) {
        distarray_axis_t *axis = dist->axes[iax];
        (*axis->global_to_rank_coords)(axis, n, global, rank_coords);
    }
}
