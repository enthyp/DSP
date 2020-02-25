import numpy as np
import os
from lpc.coding import Encoder, Decoder
from lpc.utils import get_audio
from examples.utils import butter_lowpass_filter

fs = 8000
bt_min_f, bt_max_f = 50, 400
encoder = Encoder(240, 180, 10, bt_min_f, bt_max_f, fs)
decoder = Decoder(180, 10)

audio_dir = os.path.join('/home/jlanecki/AGH/DSP/lpc-10/audio')
audio = get_audio(os.path.join(audio_dir, 'a.wav'))
left = audio[:, 0].astype(np.double)
a_lp = butter_lowpass_filter(left, 3500, sample_rate=fs)

enc = encoder.encode(a_lp)
dec = decoder.decode(enc)
