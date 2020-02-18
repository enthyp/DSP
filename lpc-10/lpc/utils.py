import sounddevice as sd
from scipy.io.wavfile import read, write

__all__ = ['dump_audio', 'get_audio', 'record_audio']


default_rate = 44100


def record_audio(duration, sample_rate=default_rate):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()
    return audio


def dump_audio(audio, filepath, sample_rate=default_rate):
    write(filepath, sample_rate, audio)


def get_audio(filepath):
    _, audio = read(filepath)
    return audio
