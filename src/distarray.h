#ifndef _e8573ca9_e599_4f27_a05f_a46e6864ab8f
#define _e8573ca9_e599_4f27_a05f_a46e6864ab8f

#include <stddef.h>
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
int distarray_grid_coords_to_rank(distarray_grid_t *grid, int *coords);

/* distarray_axis_t "base class" */
typedef struct _distarray_axis_t {
    void (*local_to_global)(struct _distarray_axis_t *restrict self,
                            ptrdiff_t n,
                            ptrdiff_t *restrict local,
                            ptrdiff_t *restrict global);
    void (*range_to_global)(struct _distarray_axis_t *restrict self,
                            int grid_coord, ptrdiff_t start, ptrdiff_t stop,
                            ptrdiff_t *restrict global);
    void (*global_to_rank_coords)(struct _distarray_axis_t *restrict self,
                                  ptrdiff_t n,
                                  ptrdiff_t *restrict global,
                                  int *restrict rank_coords);
    ptrdiff_t (*get_local_length)(struct _distarray_axis_t *self, int grid_coord);
    void (*destructor)(struct _distarray_axis_t *self);
} distarray_axis_t;


/* planning and execution */
typedef struct {
    distarray_grid_t *grid;
    distarray_axis_t **axes;
    ptrdiff_t *strides;
    ptrdiff_t *shape;
    int axes_added;
    void *_trailing_data[0];    
} distarray_distribution_t;

distarray_distribution_t *distarray_create_distribution(distarray_grid_t *grid);
void distarray_destroy_distribution(distarray_distribution_t *dist);
int distarray_distribution_adopt_axis(distarray_distribution_t *dist, ptrdiff_t len,
                                      ptrdiff_t stride, distarray_axis_t *axis);

typedef struct {
    distarray_distribution_t *from;
    distarray_distribution_t *to;
    void *_trailing_data[0];
} distarray_plan_t;

void distarray_send_counts(distarray_distribution_t *from_dist,
                           distarray_distribution_t *to_dist,
                           int from_rank, int comm_size, int *counts);
distarray_plan_t *distarray_plan_redistribution(distarray_distribution_t *from_dist,
                                                distarray_distribution_t *to_dist);
void distarray_destroy_plan(distarray_plan_t *plan);
int distarray_execute(distarray_plan_t *plan, void *from_buf, void *to_buf);


/* constructors of various axes */
int distarray_add_chunked_axis(distarray_distribution_t *dist, ptrdiff_t len,
                               ptrdiff_t stride, ptrdiff_t *offsets);



/* lower-level utilities, primarily for testing */

/* local, global are 2D arrays indexed by [n * iaxis + icoord]; 0<=icoord<n */

void distarray_local_shape(distarray_distribution_t *dist, int rank, ptrdiff_t *shape);

void distarray_local_to_global(distarray_distribution_t *dist, ptrdiff_t n,
                               ptrdiff_t *restrict local, ptrdiff_t *restrict global);

void distarray_global_to_rank_coords(distarray_distribution_t *dist, ptrdiff_t n,
                                     ptrdiff_t *restrict global, int *restrict rank_coords);

#endif
