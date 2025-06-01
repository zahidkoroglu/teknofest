import numpy as np
import scipy.signal as sig
import time
import sys

sys.path.append("/home/gokboru/Desktop/all_files/mic_array")
from mic_array import MicArray

fs = 16000
lowcut = 100.0
highcut = 7900.0
nyq = fs / 2.0
numtaps = 2047

fir = sig.firwin(
    numtaps,
    [lowcut / nyq, highcut / nyq],
    pass_zero=False,
    window="hamming"
)

threshold = 0.5  # Çok düşük enerji için bile göster
fft_window = 2048

def dominant_freq(signal, fs):
    if len(signal) < fft_window:
        return 0.0
    y = signal[-fft_window:]
    y = y - np.mean(y)
    Y = np.fft.rfft(y * np.hanning(len(y)))
    freqs = np.fft.rfftfreq(len(y), d=1/fs)
    mag = np.abs(Y)
    idx = np.argmax(mag)
    if mag[idx] < 1e-5:
        return 0.0
    return freqs[idx]

print(f"GENİŞ BANTTA DOA: {lowcut:.1f}–{highcut:.1f} Hz")

with MicArray(rate=fs, channels=4, chunk_size=fs//4) as mic:
    print("Test için hoparlörden 3000 Hz sinüs oynat. (Mikrofonun önüne getir)")
    for frames in mic.read_chunks():
        N = len(frames) // 4
        mic_signals = np.reshape(frames, (N, 4)).T

        filtered = np.zeros_like(mic_signals)
        for i in range(4):
            filtered[i] = sig.lfilter(fir, 1.0, mic_signals[i])

        energy = np.sqrt(np.mean(filtered[0] ** 2))
        if np.isnan(energy):
            energy = 0.0

        freq = dominant_freq(filtered[0], fs)

        if energy > threshold:
            filtered_frames = filtered.T.reshape(-1)
            angle = mic.get_direction(filtered_frames)
            if angle is not None:
                print(f"Açı: {angle:6.1f}°, Enerji: {energy:.1f}, Dominant Frekans: {freq:.1f} Hz")
            else:
                print(f"DOA bulunamadı | Enerji: {energy:.1f}, Frekans: {freq:.1f} Hz")
        else:
            print(f"Ses yok | Enerji: {energy:.1f}, Frekans: {freq:.1f} Hz")

        time.sleep(0.01)


