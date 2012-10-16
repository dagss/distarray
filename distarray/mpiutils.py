from __future__ import print_function

import sys
import contextlib
from mpi4py.MPI import Comm

@contextlib.contextmanager
def wait_for_turn(comm):
    assert isinstance(comm, Comm)
    rank = comm.Get_rank()
    exc_info = None
    for i in range(comm.Get_size()):
        if i == rank:
            try:
                yield
            except:
                exc_info = sys.exc_info()
        comm.Barrier()
    # Raise any exception
    if exc_info is not None and comm.Get_rank() == 0:
        raise exc_info[0], exc_info[1], exc_info[2]

def print_in_turn(comm, *args):
    assert isinstance(comm, Comm)
    with wait_for_turn(comm):
        print('%d:' % comm.Get_rank(), *args)


mprint = print_in_turn
