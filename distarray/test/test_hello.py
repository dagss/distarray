from .common import *

import distarray.hello

# run with, e.g., mpiexec -np 10 python runtests.py --nocapture

@mpi(4)
def test_hello(comm):
    print_in_turn(comm, 'hello')
    if comm.Get_rank() == 2:
        1/0
    assert 'Hello World!' == distarray.hello.hello()

