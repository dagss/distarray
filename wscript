#./waf-light --tools=compat15,swig,fc,compiler_fc,fc_config,fc_scan,gfortran,g95,ifort,gccdeps;

import os
from textwrap import dedent

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c')
    opt.load('compiler_fc')
    opt.load('python')
    opt.load('inplace', tooldir='tools')

def configure(conf):
    conf.add_os_flags('PATH')
    conf.add_os_flags('PYTHON')
    conf.add_os_flags('PYTHONPATH')
    conf.add_os_flags('INCLUDES')
    conf.add_os_flags('LIB')
    conf.add_os_flags('LIBPATH')
    conf.add_os_flags('STLIB')
    conf.add_os_flags('STLIBPATH')
    conf.add_os_flags('CFLAGS')
    conf.add_os_flags('LINKFLAGS')
    conf.add_os_flags('CYTHONFLAGS')

    conf.load('compiler_c')
    conf.load('compiler_fc')

    conf.load('python')
    conf.check_python_version((2,7))
    conf.check_python_headers()

    conf.check_tool('numpy', tooldir='tools')
    conf.check_numpy_version(minver=(1,5))
    conf.check_tool('cython', tooldir='tools')
    conf.check_cython_version(minver=(0,11,1))
    conf.check_tool('inplace', tooldir='tools')

    conf.check_mpi4py()

    conf.env.CFLAGS_C99 = ['-std=c99']


def build(bld):
    #
    # Main shared library
    #
    bld(target='distarray',
        source=['src/distarray.c'],
        includes=['src'],
        use='C99',
        features='c cshlib')

    bld(source=(['distarray/_distarray.pyx']),
        includes=['src'],
        target='_distarray',
        use='NUMPY MPI4PY distarray',
        features='c pyext cshlib')


from waflib.Configure import conf
from os.path import join as pjoin
from waflib import TaskGen

@conf
def check_mpi4py(conf):
    conf.start_msg("Checking mpi4py includes")
    (mpi4py_include,) = conf.get_python_variables(
            ['mpi4py.get_include()'], ['import mpi4py'])
    conf.env.INCLUDES_MPI4PY = mpi4py_include
    conf.end_msg('ok (%s)' % mpi4py_include)


# vim:ft=python
