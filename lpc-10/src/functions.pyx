import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef amdf(double[:] data, double[:] amdf_arr):
    cdef int i, k
    cdef float acc

    for k in range(data.shape[0]):
        acc = 0
        for i in range(data.shape[0] - k):
            acc += fabs(data[i] - data[i + k])
        amdf_arr[k] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef auto_corr(double[:] data, double[:] corr_arr):
    cdef int i, k
    cdef float acc

    for k in range(data.shape[0]):
        acc = 0
        for i in range(data.shape[0] - k):
            acc += data[i] * data[i + k]
        corr_arr[k] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cepstrum(double[:] data, double[:] cep_arr):
    pass
