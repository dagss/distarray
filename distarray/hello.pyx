

cdef extern from "chello.h":
    void say_hello(char *output, size_t maxlen)


def hello():
    cdef char buffer[20]
    say_hello(buffer, 20)
    return buffer

    
