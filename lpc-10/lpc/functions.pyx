import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void amdf(double[:] data, double[:] amdf_arr, int data_len):
    cdef int i, k
    cdef float acc

    for k in range(data.shape[0]):
        acc = 0
        for i in range(data.shape[0] - k):
            acc += fabs(data[i] - data[i + k])
        amdf_arr[k] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void auto_corr(double[:] data, double[:] corr_arr, int data_len):
    cdef int i, k
    cdef float acc

    for k in range(data_len):
        acc = 0
        for i in range(data_len - k):
            acc += data[i] * data[i + k]
        corr_arr[k] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cepstrum(double[:] data, double[:] cep_arr, int data_len):
    pass
