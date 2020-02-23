cdef class FIRFilter:
    """Generic base class for finite impulse response filters."""

    cdef int kernel_length
    cdef double [:] buffer, kernel
    cpdef run(self, double[:] samples, double[:] filtered, int samples_len)


cdef class LPFilter(FIRFilter):
    """Windowed-sinc low-pass filter."""

    cdef double fc


cdef class PreempFilter(FIRFilter):
    """Simple pre-emphasis filter."""
