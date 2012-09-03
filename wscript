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

def build(bld):
    #
    # Main shared library
    #
    #bld(target='wavemoth',
    #    source=['src/butterfly.c.in'],
    #    includes=['src'],
    #    use='C99',
    #    features='c cshlib')

    #bld(source=(['wavemoth/butterfly.pyx']),
    #    includes=['src'],
    #    target='butterfly',
    #    use='NUMPY fcshlib wavemoth',
    #    features='c pyext cshlib')


# vim:ft=python
