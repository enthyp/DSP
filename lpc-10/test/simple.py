import numpy as np
import os
from lpc.coding import Encoder
from lpc.utils import get_audio

fs = 44100
bt_min_f, bt_max_f = 50, 400
encoder = Encoder(240, 180, 50, 400)

audio_dir = os.path.join('/home/jlanecki/AGH/DSP/lpc-10/audio')

audio = get_audio(os.path.join(audio_dir, 'a.wav'))
left = audio[:, 0].astype(np.double)
enc = encoder.encode(left, fs)
