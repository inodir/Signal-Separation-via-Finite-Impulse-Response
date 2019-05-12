import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
import contextlib


def fir_low_pass(samples, fs, fL, N, outputType):

    fL = fL / fs
    h = np.sinc(2 * fL * (np.arange(N) - (N - 1) / 2.))       # Compute sinc filter.
    h *= np.hamming(N)		 								  # Apply window.
    h /= np.sum(h)		 									  # Normalize to get unity gain.
    s = np.convolve(samples, h).astype(outputType)            # Applying the filter to a signal s can be as simple as writing
    return s

def fir_high_pass(samples, fs, fH, N, outputType):

    fH = fH / fs

    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.))  	  # Compute sinc filter.
    h *= np.hamming(N)  									  # Apply window.
    h /= np.sum(h)	 										  # Normalize to get unity gain.
    h = -h   	 											  # Create a high-pass filter from the low-pass filter through spectral inversion.
    h[int((N - 1) / 2)] += 1	
    s = np.convolve(samples, h).astype(outputType)   		  # Applying the filter to a signal s can be as simple as writing
    return s

def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)
    if interleaved:
        channels.shape = (n_frames, n_channels)				  # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels = channels.T
    else:
        channels.shape = (n_channels, n_frames)				  # channels are not interleaved. All samples from channel M occur before all samples from channel M-1

    return channels

def get_start_end_frames(nFrames, sampleRate, tStart=None, tEnd=None): 

    if tStart and tStart*sampleRate<nFrames:
        start = tStart*sampleRate
    else:
        start = 0

    if tEnd and tEnd*sampleRate<nFrames and tEnd*sampleRate>start:
        end = tEnd*sampleRate
    else:
        end = nFrames

    return (start,end,end-start)

def extract_audio(fname, tStart=None, tEnd=None):
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        startFrame, endFrame, segFrames = get_start_end_frames(nFrames, sampleRate, tStart, tEnd)
        spf.setpos(startFrame)                     						 # Extract Raw Audio from multi-channel Wav File
        sig = spf.readframes(segFrames)
        spf.close()

        channels = interpret_wav(sig, segFrames, nChannels, ampWidth, True)
        return (channels, nChannels, sampleRate, ampWidth, nFrames)

def convert_to_mono (channels, nChannels, outputType):      
    if nChannels == 2:
        samples = np.mean(np.array([channels[0], channels[1]]), axis=0)  # Convert to mono
    else:
        samples = channels[0]

    return samples.astype(outputType)

def plot_specgram(samples, sampleRate, tStart=None, tEnd=None):
    plt.figure(figsize=(20,10))
    plt.specgram(samples, Fs=sampleRate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(tStart,tEnd))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def plot_audio_samples(title, samples, sampleRate, tStart=None, tEnd=None): #+++++++++++++++++++++
    if not tStart:
        tStart = 0

    if not tEnd or tStart>tEnd:
        tEnd = len(samples)/sampleRate

    f, axarr = plt.subplots(2, sharex=True, figsize=(20,10))
    axarr[0].set_title(title)
    axarr[0].plot(np.linspace(tStart, tEnd, len(samples)), samples)
    axarr[1].specgram(samples, Fs=sampleRate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(tStart,tEnd))
 
    axarr[0].set_ylabel('Amplitude')
    axarr[1].set_ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

tStart=0
tEnd=11

channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('Team-6-Project2_v2_TEST.wav', tStart, tEnd)
samples = convert_to_mono(channels, nChannels, np.int16)

lp_samples_filtered = fir_low_pass(samples, sampleRate, 600, 461, np.int16)               # First pass
lp_samples_filtered = fir_low_pass(lp_samples_filtered, sampleRate, 500, 461, np.int16)   # Second pass

hp_samples_filtered = fir_high_pass(samples, sampleRate, 15100, 461, np.int16)             # First pass
hp_samples_filtered = fir_high_pass(hp_samples_filtered, sampleRate, 15000, 461, np.int16) # Second pass

samples_filtered = np.mean(np.array([lp_samples_filtered, hp_samples_filtered]), axis=0).astype(np.int16)
wavfile.write('filtered.wav', sampleRate, samples_filtered)
plot_audio_samples("Signal Separation via Finite Impulse Response", samples, sampleRate, tStart, tEnd)
