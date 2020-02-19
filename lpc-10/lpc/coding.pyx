cimport cython
import numpy as np
cimport numpy as np
from .functions cimport auto_corr, amdf, cepstrum

ctypedef void (*bt_function)(double[:] data, double[:] result)

bt_fmap = {
    'correlation': 0,
    'amdf': 1,
    'cepstrum': 2
}

cdef bt_function bt_functions[3]
bt_functions[0] = auto_corr
bt_functions[1] = amdf
bt_functions[2] = cepstrum


cdef class Encoder:
    cdef int w_len, w_step
    cdef double bt_min_f, bt_max_f
    cdef bt_function bt_fun

    def __init__(self, int w_len, int w_step, double bt_min_f, double bt_max_f, bt_fun='correlation'):
        ind = bt_fmap[bt_fun]
        self.bt_fun = bt_functions[ind]
        self.w_len = w_len
        self.w_step = w_step
        self.bt_min_f = bt_min_f
        self.bt_max_f = bt_max_f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef encode(self, double[:] audio, double fs):
        cdef int i = 0, bt_min_s, bt_max_s, tmp
        cdef double[:] window_metrics
        cdef double[:, :] out_frames

        bt_max_s = <int>(fs // self.bt_min_f)
        bt_min_s = <int>(fs // self.bt_max_f)

        tmp = audio.shape[0] // self.w_step
        out_frames = np.empty((tmp, 2), dtype=np.double)
        window_metrics = np.empty((self.w_len,), dtype=np.double)

        while i < audio.shape[0]:
            # Get base tone characteristic first.
            self.bt_fun(audio[i:i + self.w_len], window_metrics)
            out_frames[i // self.w_step][0] = self.find_period(window_metrics, bt_min_s, bt_max_s, fs)

            i += self.w_step
        return out_frames

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef find_period(self, double[:] metric, int bt_min_s, int bt_max_s, double fs):
        cdef int i = 0, max_ind = bt_min_s

        for i in range(bt_min_s, bt_max_s):
            if metric[i] > metric[max_ind]:
                max_ind = i

        if metric[bt_min_s + max_ind] > 0.35 * metric[0]:
            return (max_ind + bt_min_s) / fs
        else:
            return 0.0  # well, works most of the time...



cdef class Decoder:
    def __init__(self):
        pass

    def decode(self):
        pass
