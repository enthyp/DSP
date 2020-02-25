# LPC-10
Sketchy implementation of Linear Predictive Coding method, 
applied to speech compression and synthesis.

The quality is definitely poor, the fewer vocal filter coefficients 
the worse, but it works! Decompressed speech is audible and that's amazing.

### Potential upgrades
 * Use other (AMDF/cepstral) method to estimate base tone frequency
 * Use vocal filter coefficient smoothing between frames (LSP/LSF polynomials)
 * ...?

### Description (sort of)
Working principle is described in _examples_:
 * `speech_recording` - that's just for the purpose of recording
   some voiced/unvoiced sounds
 * `vocal_source_estimation` - here we explore some methods that
   distinguish voiced sounds from unvoiced ones and estimate the
   base tone frequency for the voiced ones
 * `vocal_tract_estimation` - here we estimate parameters of the
   vocal filter for a single 30ms time window of recorded sounds 
 * in `pipeline` we put together all the steps from the above two
   and perform actual encoding and decoding for a single time window
 * ... and in `compression` we use Cython encoder/decoder to verify
   quality of speech compression on a longer (~15 seconds) audio
   recording.
 

What's where:
 * `functions.pyx` contains functions one can use to estimate
   the base tone frequency. No need for that to be in a separate
   file, that's just to see how Cython imports are done
 * `fir.pyx` contains basic finite impulse response (FIR) filters.
   These are used for pre-emphasis and 900Hz low-pass before base
   tone estimation
 * `coding.pyx` - encoder and decoder described above
   
To run the examples you need `jupyter` installed.

You also need to install `lpc` package in your Jupyter 
environment (best use a new conda environment
 with `nb_conda_kernels` installed). 
For that purpose just run `make` in the main project directory.
