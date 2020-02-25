cimport cython
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cl

from libc.math cimport cos, pi
from libc.stdlib cimport rand, malloc, free
cdef extern from "stdlib.h":
    int RAND_MAX

from lpc.functions cimport auto_corr, amdf, cepstrum
from lpc.fir cimport *


ctypedef void (*bt_function)(double[:] data, double[:] result, int data_len)

cdef bt_function bt_functions[3]
bt_functions[0] = auto_corr
bt_functions[1] = amdf
bt_functions[2] = cepstrum


cdef class Encoder:
    cdef int w_len, w_step, n_coef
    cdef double bt_min_f, bt_max_f, fs
    cdef double[:] window
    cdef FIRFilter preemp
    cdef FIRFilter lp_filter

    def __cinit__(self, int w_len, int w_step, int n_coef, double bt_min_f, double bt_max_f, double fs):
        self.w_len = w_len
        self.w_step = w_step
        self.n_coef = n_coef
        self.bt_min_f = bt_min_f
        self.bt_max_f = bt_max_f
        self.fs = fs
        self.preemp = PreempFilter(2)
        self.lp_filter = LPFilter(127, 900 / fs)
        self.window = np.empty((w_len,), dtype=np.double)

    def __init__(self, int w_len, int w_step, int n_coef, double bt_min_f, double bt_max_f, double fs):
        self.prepare_window()

    cdef prepare_window(self):
        # Hamming window.
        cdef int i
        cdef double arg

        for i in range(self.w_len):
            arg = (i - self.w_len / 2) / (self.w_len / 2)
            self.window[i] = 0.54 + 0.46 * cos(pi * arg)

    # @cython.boundscheck(False)
    cpdef encode(self, double[:] audio):
        cdef int i = 0, j, frame_no, bt_min_s, bt_max_s, tmp
        cdef double base_period
        cdef int[:] i_piv
        cdef double[:] preemp, filtered, coef, corr
        cdef double[:, :] r_matrix, out_frames

        cdef int l_work=-1, n=self.n_coef, n_rhs=1, info
        cdef double wk_opt
        cdef char* uplo = 'U'
        cdef double *work_arr

        bt_max_s = <int>(self.fs // self.bt_min_f)
        bt_min_s = <int>(self.fs // self.bt_max_f)

        # Results.
        # layout: [T, a_1, a_2, ..., a_{n_coef}, G]
        tmp = <int>(audio.shape[0] // self.w_step)
        if  self.w_step * (tmp - 1) + self.w_len > audio.shape[0]:
            tmp -= 1
        out_frames = np.empty((tmp, 2 + self.n_coef), dtype=np.double)

        # Steps.
        preemp = np.empty((self.w_len,), dtype=np.double)
        filtered = np.empty((self.w_len,), dtype=np.double)
        corr = np.empty((self.w_len,), dtype=np.double)
        coef = np.empty((self.n_coef + 1,), dtype=np.double)

        # Temporary.
        r_matrix = np.empty((self.n_coef, self.n_coef), dtype=np.double)
        i_piv = np.empty((self.n_coef,), dtype=np.int32)

        # Alloc memory for LAPACK DSYSV routine o.O
        cl.dsysv(uplo, &n, &n_rhs, &r_matrix[0, 0], &n, &i_piv[0], &coef[0], &n, &wk_opt, &l_work, &info)
        l_work = <int>wk_opt
        work_arr = <double *> malloc(l_work * sizeof(double))

        while i + self.w_len <= audio.shape[0]:
            frame_no = <int>(i // self.w_step)

            # Preemphasis.
            self.preemp.run(audio[i:], preemp, self.w_len)

            # Hamming multiplication.
            for j in range(self.w_len):
                preemp[j] *= self.window[j]

            # Get base tone characteristic first.
            # Low-pass.
            self.lp_filter.run(preemp, filtered, self.w_len)

            # TODO: thresholds?
            # Find base period.
            auto_corr(filtered, corr, self.w_len)
            base_period = self.find_period(corr, bt_min_s, bt_max_s)
            out_frames[frame_no][0] = base_period

            # Next, estimate vocal tract filter params.
            tmp = self.n_coef
            auto_corr(preemp, corr, self.w_len)
            self.find_coef(corr, coef, r_matrix, i_piv, tmp, l_work, work_arr)

            for j in range(self.n_coef + 1):
                out_frames[frame_no][j + 1] = coef[j]
            i += self.w_step

        free(work_arr)
        return np.array(out_frames)

    # @cython.boundscheck(False)
    cdef find_coef(self, double[:] corr, double[:] coef, double[:, :] r_matrix, int[:] i_piv, int n_coef, int l_work, double* work_arr):
        cdef int i, j, n=n_coef, n_rhs=1, info
        cdef char* uplo = 'U'

        # Prepare RHS (tb solution).
        for i in range(n_coef):
            coef[i] = -corr[i + 1]

        # Prepare linear system's matrix.
        for i in range(n):
            for j in range(i, n):
                r_matrix[i, j] = corr[j - i]
            for j in range(0, i):
                r_matrix[i, j] = corr[i - j]

        # Solve the system.
        cl.dsysv(uplo, &n, &n_rhs, &r_matrix[0, 0], &n, &i_piv[0], &coef[0], &n, work_arr, &l_work, &info)

        coef[n_coef] = 0
        for i in range(1, n_coef + 1):
            coef[n_coef] += corr[i] * coef[i - 1]
        coef[n_coef] += corr[0]

    @cython.boundscheck(False)
    cdef double find_period(self, double[:] corr, int bt_min_s, int bt_max_s):
        cdef int i = 0, max_ind = bt_min_s

        for i in range(bt_min_s, bt_max_s):
            if corr[i] > corr[max_ind]:
                max_ind = i

        if corr[max_ind] > 0.35 * corr[0]:
            return <double>max_ind
        else:
            return 0.0


cdef class Decoder:
    cdef int w_step, n_coef

    def __cinit__(self, int w_step, int n_coef):
        self.w_step = w_step
        self.n_coef = n_coef

    def __init__(self, int w_step, int n_coef):
        pass

    cpdef decode(self, double[:, :] data):
        cdef int i, j, k, base_period_s
        cdef double impulse, tmp, tmp2, acc, deemp = 0
        cdef double[:] buffer, out_samples

        buffer = np.zeros((self.n_coef,), dtype=np.double)
        out_samples = np.empty((data.shape[0] * self.w_step,), dtype=np.double)

        for i in range(data.shape[0]):
            base_period_s = <int>data[i, 0]

            for j in range(self.w_step):
                # Get excitation value for voiced/unvoiced case.
                if base_period_s == 0:
                    # Unvoiced.
                    impulse = 2 * (rand() / <double>RAND_MAX - 0.5)
                else:
                    # Voiced.
                    if j % base_period_s == 0:
                        impulse = 1.0
                    else:
                        impulse = 0.0

                # Calculate filter output.
                acc = data[i, self.n_coef + 1] * impulse
                for k in range(1, self.n_coef + 1):
                    acc -= buffer[k - 1] * data[i, k]

                # Shift the buffer.
                tmp = buffer[0]
                buffer[0] = acc
                for k in range(1, self.n_coef):
                    tmp2 = buffer[k]
                    buffer[k] = tmp
                    tmp = tmp2

                # De-emphasis.
                deemp = acc + 0.9375 * deemp
                out_samples[i * self.w_step + j] = deemp

        return np.array(out_samples)
