import os
import glob
import numpy as np
from scipy.io import wavfile
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_base = "ToyADMOS"
subdataset = "ToyConveyor"
#case_num = "case1"
#ch_num = "ch1"
target_fs = 16000

normal_dir = os.path.join(dataset_base, subdataset, "NormalSound_IND")
anomaly_dir = os.path.join(dataset_base, subdataset, "AnomalousSound_IND")

def wavread(fn):
    fs, data = wavfile.read(fn)
    data = data.astype(np.float32) / 2**15
    return data, fs

def load_wavs_from_dir(wav_dir, target_fs):
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    signals = []
    filenames = []
    print(f'Loading {len(wav_files)} files from {wav_dir} ...')
    for fn in tqdm(wav_files):
        signal, fs = wavread(fn)
        if fs != target_fs:
            signal = librosa.resample(y=signal, orig_sr=fs, target_sr=target_fs)
        signals.append(signal)
        filenames.append(os.path.basename(fn))
    return signals, filenames

#Wczytanie plików wav
S_normal, fn_normal = load_wavs_from_dir(normal_dir, target_fs)
S_anomaly, fn_anomaly = load_wavs_from_dir(anomaly_dir, target_fs)

print(f'Loaded {len(S_normal)} normal samples')
print(f'Loaded {len(S_anomaly)} anomalous samples')


# Wydzielenie sygnału do wyświetlenia
signal = S_normal[0]

# Stereo -> Mono
if signal.ndim > 1:
    signal = signal.mean(axis=1)

signal_short = signal[:5*target_fs]

# FFT
D = librosa.stft(signal_short, n_fft=1024, hop_length=256)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Wyświetlenie spektrogramu
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db,
                         sr=target_fs,
                         x_axis='time',
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Example Spectrogram')
plt.tight_layout()
plt.show()
