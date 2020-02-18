from functions import *

ctypedef void (bt_function)(double *data, double *result)

bt_functions = {
    'correlation': auto_corr,
    'amdf': amdf,
    'cepstrum': cepstrum
}

cdef class Encoder:
    cdef bt_function bt_fun

    def __init__(self, bt_fun='correlation'):  # TODO: fs, window length, window step, base tone limit frequencies...
        self.bt_fun = bt_functions[bt_fun]

    def encode(self, audio):
        pass


cdef class Decoder:
    def __init__(self):
        pass

    def decode(self):
        pass
