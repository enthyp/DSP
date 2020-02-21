cimport cython
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cl

from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

from .functions cimport auto_corr, amdf, cepstrum


ctypedef void (*bt_function)(double[:] data, double[:] result)

cdef bt_function bt_functions[3]
bt_functions[0] = auto_corr
bt_functions[1] = amdf
bt_functions[2] = cepstrum


cdef class Encoder:
    cdef int w_len, w_step, n_coef
    cdef double bt_min_f, bt_max_f

    def __init__(self, int w_len, int w_step, int n_coef, double bt_min_f, double bt_max_f):
        self.w_len = w_len
        self.w_step = w_step
        self.n_coef = n_coef
        self.bt_min_f = bt_min_f
        self.bt_max_f = bt_max_f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef encode(self, double[:] audio, double fs):
        cdef int j, i = 0, bt_min_s, bt_max_s, tmp
        cdef double base_period
        cdef int[:] i_piv
        cdef double[:] coef, corr
        cdef double[:, :] r_matrix, out_frames

        bt_max_s = <int>(fs // self.bt_min_f)
        bt_min_s = <int>(fs // self.bt_max_f)

        r_matrix = np.empty((self.n_coef, self.n_coef), dtype=np.double)
        out_frames = np.empty((audio.shape[0] // self.w_step, 2 + self.n_coef), dtype=np.double)
        corr = np.empty((self.w_len,), dtype=np.double)
        coef = np.empty((self.n_coef + 1,), dtype=np.double)
        i_piv = np.empty((self.n_coef,), dtype=np.int32)

        while i < audio.shape[0]:
            # TODO:
            #  - preemphasis
            #  - Hamming
            #  - low-pass 900Hz for base tone
            #  - thresholding!

            # Get base tone characteristic first.
            auto_corr(audio[i:i + self.w_len], corr)
            base_period = self.find_period(corr[i: i + self.w_len], bt_min_s, bt_max_s, fs)
            out_frames[i // self.w_step][0] = base_period

            # Next, estimate vocal tract filter params.
            tmp = self.n_coef
            self.find_coef(corr, coef, r_matrix, i_piv, tmp)

            for j in range(self.n_coef + 1):
                out_frames[i // self.w_step][j + 1] = coef[j]
            i += self.w_step

        return np.array(out_frames)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double find_coef(self, double[:] corr, double[:] coef, double[:, :] r_matrix, int[:] dump, int n_coef):
        cdef int i, j, n=n_coef, n_rhs=1, info

        # Prepare RHS (tb solution).
        for i in range(n):
            coef[i] = -corr[i + 1]

        # Prepare linear system's matrix.
        for i in range(n):
            for j in range(i, n):
                r_matrix[i, j] = corr[j - i]
            for j in range(i, 0, -1):
                r_matrix[i, i - j] = corr[j]

        # Solve the system.
        cl.dgesv(&n, &n_rhs, &r_matrix[0, 0], &n, &dump[0], &coef[0], &n, &info)

        # Solve for G.
        for i in range(1, n_coef + 1):
            coef[n_coef] += corr[i] * coef[i - 1]
        coef[n_coef] += corr[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double find_period(self, double[:] corr, int bt_min_s, int bt_max_s, double fs):
        cdef int i = 0, max_ind = bt_min_s

        for i in range(bt_min_s, bt_max_s):
            if corr[i] > corr[max_ind]:
                max_ind = i

        if corr[max_ind] > 0.35 * corr[0]:
            return max_ind / fs
        else:
            return 0.0  # well, works most of the time...


cdef class Decoder:
    cdef int w_step, n_coef

    def __init__(self, int w_step, int n_coef):
        self.w_step = w_step
        self.n_coef = n_coef

    def decode(self, data):
        cdef int i, j, k
        cdef double rnd, acc = 0, tmp, tmp2
        cdef double[:] buffer, out_samples

        buffer = np.zeros((self.n_coef,), dtype=np.double)
        out_samples = np.empty((data.shape[0] * self.w_step,), dtype=np.double)

        for i in range(data.shape[0]):
            if data[i][0] < 1e-5:
                 # Unvoiced.
                for j in range(self.w_step):
                    tmp = buffer[0]
                    for k in range(1, self.n_coef):
                        tmp2 = buffer[k]
                        buffer[k] = tmp
                        tmp = tmp2

                    buffer[0] = 2 * (rand() / double(RAND_MAX) - 0.5)

                    acc = 0
                    for k in range(self.n_coef):
                        acc += buffer[i] * data[i][k]

                    out_samples[i * self.w_step + j] = acc
            else:
                # Voiced.
                pass

        return np.array(out_samples)
