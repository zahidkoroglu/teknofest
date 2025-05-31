import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt

# --- SESİ YÜKLE ---
fs, x = wavfile.read("system.wav")     # input.wav dosyanı aynı klasöre koy

# Mono'ya çevir (stereo ise)
if x.ndim > 1:
    x = x.mean(axis=1)
x = x.astype(np.float32)

# --- FİLTRE PARAMETRELERİ ---
center     = 500.0              # Geçirmek istediğin merkez frekans
bandwidth  = 20.0               # Bant genişliği (ör: ±10 Hz)
lowcut     = center - bandwidth/2
highcut    = center + bandwidth/2
nyq        = fs / 2.0

# --- FIR DARBANT GEÇİREN FİLTRE TASARLA ---
numtaps = 4097                  # 4097 tap ile oldukça dar ve temiz bant elde edilir
fir = sig.firwin(
        numtaps,
        [lowcut/nyq, highcut/nyq],
        pass_zero=False,
        window="hamming")

# --- FİLTRE UYGULA (sıfır faz kayması ile) ---
y = sig.filtfilt(fir, 1.0, x)

# --- PSD ANALİZİ (Öncesi ve Sonrası) ---
f1, Pxx1 = sig.welch(x, fs, nperseg=4096)
f2, Pxx2 = sig.welch(y, fs, nperseg=4096)

plt.figure(figsize=(12, 5))
plt.semilogy(f1, Pxx1, label="Filtresiz Ses", alpha=0.8)
plt.semilogy(f2, Pxx2, label="Filtreli Ses (500 Hz Bandı)", color='orange', alpha=0.8)
plt.title("PSD Karşılaştırması (Öncesi ve Sonrası)")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Güç Yoğunluğu [V**2/Hz]")
plt.grid()
plt.legend()
plt.tight_layout()

# --- SONUCU WAV OLARAK KAYDET ---
y = y / np.max(np.abs(y))       # Normalizasyon
wavfile.write("output_500hz_2.wav", fs, y.astype(np.float32))

# --- FİLTRENİN FREKANS CEVABINI GÖSTER (opsiyonel) ---
w, h = sig.freqz(fir, worN=8192, fs=fs)
plt.figure(figsize=(10, 4))
plt.plot(w, 20*np.log10(np.abs(h)))
plt.title("500 Hz Dar Bant Geçiren FIR Filtre Frekans Cevabı")
plt.xlabel("Frekans [Hz]")
plt.ylabel("Kazanç [dB]")
plt.grid()
plt.tight_layout()

plt.show()
