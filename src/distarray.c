#include "distarray.h"

#include <assert.h>
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

int distarray_grid_coords_to_rank(distarray_grid_t *grid, int *coords) {
    int i = 0;
    for (int iax = 0; iax != grid->ndim; ++iax) {
        i += grid->strides[iax] * coords[iax];
    }
    return grid->ranks[i];
}


/*
distribution
*/
distarray_distribution_t *distarray_create_distribution(distarray_grid_t *grid) {
    int ndim = grid->ndim;
    distarray_distribution_t *dist = malloc(sizeof(distarray_distribution_t) + /* struct */
                                            sizeof(void*) * ndim + /* axes array */
                                            sizeof(ptrdiff_t) * ndim * 2); /* stride&shape array */
    dist->grid = grid;
    dist->axes_added = 0;

    char *head = (char*)dist + sizeof(distarray_distribution_t);
    dist->axes = (distarray_axis_t**)head;
    head += ndim * sizeof(void*);
    dist->shape = (ptrdiff_t*)head;
    head += ndim * sizeof(ptrdiff_t);
    dist->strides = (ptrdiff_t*)head;

    memset(dist->axes, 0, sizeof(void*) * ndim);
    memset(dist->shape, 0, sizeof(ptrdiff_t) * ndim);
    memset(dist->strides, 0, sizeof(ptrdiff_t) * ndim);

    return dist;
}

void distarray_destroy_distribution(distarray_distribution_t *dist) {
    for (int i = 0; i != dist->axes_added; ++i) {
        dist->axes[i]->destructor(dist->axes[i]);
    }
    free(dist);
}

int distarray_distribution_adopt_axis(distarray_distribution_t *dist, ptrdiff_t len,
                                      ptrdiff_t stride, distarray_axis_t *axis) {
    if (dist->axes_added == dist->grid->ndim) return -1;
    dist->axes[dist->axes_added] = axis;
    dist->shape[dist->axes_added] = len;
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

static void _chunked_range_to_global(distarray_axis_t *restrict self,
                                     int grid_coord,
                                     ptrdiff_t start, ptrdiff_t stop,
                                     ptrdiff_t *restrict global_indices) {
    ptrdiff_t local_start = ((distarray_chunked_axis_t *)self)->offsets[grid_coord];
    
    for (ptrdiff_t i = start; i != stop; ++i) {
        global_indices[i - start] = local_start + i;
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

static ptrdiff_t _chunked_get_local_length(distarray_axis_t *self, int grid_coord) {
    ptrdiff_t *offsets = ((distarray_chunked_axis_t*)self)->offsets;
    return offsets[grid_coord + 1] - offsets[grid_coord];
}
                                           

int distarray_add_chunked_axis(distarray_distribution_t *dist, ptrdiff_t len,
                               ptrdiff_t stride, ptrdiff_t *offsets) {
    distarray_chunked_axis_t *axis = malloc(sizeof(distarray_chunked_axis_t));
    axis->offsets = offsets;
    axis->base.destructor = &_default_axis_destructor;
    axis->base.local_to_global = &_chunked_local_to_global;
    axis->base.range_to_global = &_chunked_range_to_global;
    axis->base.global_to_rank_coords = &_chunked_global_to_rank_coords;
    axis->base.get_local_length = &_chunked_get_local_length;

    int iaxis = dist->axes_added;
    
    int rank;
    if (MPI_Comm_rank(dist->grid->comm, &rank) != MPI_SUCCESS) return -1;
    int grid_coords[dist->grid->ndim];
    if (distarray_grid_rank_to_coords(dist->grid, rank, grid_coords) != 0) return -1;
    axis->local_start = offsets[grid_coords[iaxis]];
    axis->local_stop = offsets[grid_coords[iaxis] + 1];

    distarray_distribution_adopt_axis(dist, len, stride, (void*)axis);

    return 0;
}


/* 
   planning & execution
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

void distarray_send_counts(distarray_distribution_t *from_dist,
                           distarray_distribution_t *to_dist,
                           int from_rank, int comm_size, int *counts) {
    assert(from_dist->grid->ndim == 2); /* TODO */

    memset(counts, 0, sizeof(int) * comm_size);

    ptrdiff_t *shape = from_dist->shape;

    int ndim = from_dist->grid->ndim;
    int from_grid_coords[ndim];
    int to_grid_coords[ndim];
    ptrdiff_t from_shape[ndim];

    distarray_local_shape(from_dist, from_rank, from_shape);
    distarray_grid_rank_to_coords(from_dist->grid, from_rank, from_grid_coords);
    

    /* TODO: To conserve memory, do this in chunks rather than getting entire axes
       right away */

    ptrdiff_t *global0 = malloc(sizeof(ptrdiff_t) * from_shape[0]);
    int *grid_coord0 = malloc(sizeof(int) * from_shape[0]);
    ptrdiff_t *global1 = malloc(sizeof(ptrdiff_t) * from_shape[1]);
    int *grid_coord1 = malloc(sizeof(int) * from_shape[1]);

    (*from_dist->axes[0]->range_to_global)(from_dist->axes[0], from_grid_coords[0],
                                           0, from_shape[0], global0);
    (*to_dist->axes[0]->global_to_rank_coords)(to_dist->axes[0], 
                                               from_shape[0], global0, grid_coord0);

    (*from_dist->axes[1]->range_to_global)(from_dist->axes[1], from_grid_coords[1],
                                           0, from_shape[1], global1);
    (*to_dist->axes[1]->global_to_rank_coords)(to_dist->axes[1], 
                                               from_shape[1], global1, grid_coord1);

    distarray_grid_t *to_grid = to_dist->grid;
    int grid_coords[ndim];
    for (ptrdiff_t i0 = 0; i0 != from_shape[0]; ++i0) {
        grid_coords[0] = grid_coord0[i0];
        for (ptrdiff_t i1 = 0; i1 != from_shape[1]; ++i1) {
            grid_coords[1] = grid_coord1[i1];
            int to_rank = distarray_grid_coords_to_rank(to_grid, grid_coords);
            counts[to_rank] += 1;
        }
    }

    free(global0);
    free(global1);
    free(grid_coord0);
    free(grid_coord1);
}

int distarray_execute(distarray_plan_t *plan, void *from_buf, void *to_buf) {
    distarray_grid_t *grid = plan->from->grid;
    MPI_Comm comm = grid->comm;
    int rank, comm_size;
    if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS) return -1;
    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) return -1;

    int *sendcounts = malloc(comm_size * sizeof(int));
    int *recvcounts = malloc(comm_size * sizeof(int));

    distarray_send_counts(plan->from, plan->to, rank, comm_size, sendcounts);
    /* To find recvcounts we can simple think about the reverse redistribution */
    distarray_send_counts(plan->to, plan->from, rank, comm_size, recvcounts);

    for (int i = 0; i != comm_size; ++i) {
        printf("%d:%d:%d\n", rank, i, sendcounts[i], recvcounts[i]);
    }

    free(sendcounts);
    free(recvcounts);
}





/* utils */

void distarray_local_shape(distarray_distribution_t *dist, int rank, ptrdiff_t *shape) {
    int ndim = dist->grid->ndim;
    int grid_coord[ndim];

    distarray_grid_rank_to_coords(dist->grid, rank, grid_coord);
 
    for (int iax = 0; iax != ndim; ++iax) {
        distarray_axis_t *axis = dist->axes[iax];
        shape[iax] = (*axis->get_local_length)(axis, grid_coord[iax]);
    }
}

void distarray_local_to_global(distarray_distribution_t *dist, ptrdiff_t n,
                               ptrdiff_t *restrict local, ptrdiff_t *restrict global) {
    int ndim = dist->grid->ndim;

    for (int iax = 0; iax != ndim; ++iax) {
        distarray_axis_t *axis = dist->axes[iax];
        (*axis->local_to_global)(axis, n, local, global);
    }
}

void distarray_global_to_rank_coords(distarray_distribution_t *dist, ptrdiff_t n,
                                     ptrdiff_t *restrict global, int *restrict rank_coords) {
    int ndim = dist->grid->ndim;

    for (int iax = 0; iax != ndim; ++iax) {
        distarray_axis_t *axis = dist->axes[iax];
        (*axis->global_to_rank_coords)(axis, n, global, rank_coords);
    }
}
