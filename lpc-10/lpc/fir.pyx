cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport cos, pi, sin

__all__ = ['FIRFilter', 'LPFilter', 'PreempFilter']


cdef class FIRFilter:
    """Generic base class for finite impulse response filters."""

    def __init__(self, int kernel_length, **args):
        self._prepare_kernel()

    def _prepare_kernel(self):
        raise NotImplementedError

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef run(self, double[:] samples, double[:] filtered, int samples_len):
        cdef int i, j
        cdef double acc, tmp, tmp2

        for i in range(samples_len):
            # Shift the buffer.  # TODO: use circular buffer.
            tmp = self.buffer[0]
            self.buffer[0] = samples[i]

            for j in range(1, self.kernel_length):
                tmp2 = self.buffer[j]
                self.buffer[j] = tmp
                tmp = tmp2

            # Convolve.
            acc = 0
            for j in range(self.kernel_length):
                acc += self.buffer[j] * self.kernel[j]

            filtered[i] = acc


cdef class LPFilter(FIRFilter):
    """Windowed-sinc low-pass filter."""

    def __cinit__(self, int kernel_length, double fc):
        self.kernel_length = kernel_length
        self.buffer = np.zeros((kernel_length,), dtype=np.double)
        self.kernel = np.zeros((kernel_length,), dtype=np.double)
        self.fc = fc

    def __init__(self, int kernel_length, double fc):
        super().__init__(kernel_length)

    def _prepare_kernel(self):
        cdef int i
        cdef double s

        s = 0
        for i in range(self.kernel_length):
            if i == self.kernel_length // 2:
                self.kernel[i] = 2 * pi * self.fc
            else:
                self.kernel[i] = sin(2 * pi * self.fc * (i - self.kernel_length / 2)) / (i - self.kernel_length / 2)

            self.kernel[i] *= (0.42 - 0.5 * cos(2 * pi * i / self.kernel_length) + 0.08 * cos(4 * pi * i / self.kernel_length))
            s += self.kernel[i]

        for i in range(self.kernel_length):
            self.kernel[i] /= s


cdef class PreempFilter(FIRFilter):
    """Simple pre-emphasis filter."""

    def __cinit__(self):
        self.kernel_length = 2
        self.buffer = np.zeros((self.kernel_length,), dtype=np.double)
        self.kernel = np.zeros((self.kernel_length,), dtype=np.double)

    def _prepare_kernel(self):
        cdef int i
        cdef double s

        self.kernel[0] = 1
        self.kernel[1] = -0.9375
