cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport cos, pi, sin


cdef class LPFilter:
    """Windowed-sinc low-pass filter.."""
    
    cdef int kernel_length
    cdef double fc
    cdef double [::1] buffer, kernel

    def __init__(self, int kernel_length, double fc):
        self.fc = fc
        self.buffer = np.zeros((kernel_length,), dtype=np.double)
        self.kernel_length = kernel_length
        self.kernel = np.zeros((kernel_length,), dtype=np.double)

        self._prepare_kernel()


    cdef _prepare_kernel(self):
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)            
    cpdef run(self, double[::1] samples, double[::1] filtered, int n):
        cdef int i, j, offset
        cdef double acc, tmp, tmp2

        for i in range(n):
            # Shift the buffer.
            tmp = self.buffer[0]
            for j in range(1, self.kernel_length):
                tmp2 = self.buffer[j]
                self.buffer[j] = tmp
                tmp = tmp2

            # Convolve.
            acc = 0
            for j in range(self.kernel_length):
                acc += samples[j] * self.kernel[j]

            filtered[i] = acc
