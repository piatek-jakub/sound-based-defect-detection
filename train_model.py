import os
import glob
import numpy as np
import pandas as pd
import re
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# =========================
# Parametry
# =========================
dataset_base = "ToyADMOS dataset"
subdataset = "ToyCar"
target_fs = 48000
models_dir = "models"
labels_csv_path = "labels/ToyCar_anomay_condition.csv"

normal_dir = os.path.join(dataset_base, subdataset, "case1", "NormalSound_IND")
anomaly_dir = os.path.join(dataset_base, subdataset, "case1", "AnomalousSound_IND")

# =========================
# Funkcje
# =========================
def wavread(fn):
    fs, data = wavfile.read(fn)
    data = data.astype(np.float32) / 2**15
    return data, fs

def load_wavs_from_dir(wav_dir, target_fs):
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    signals = []
    file_names = []
    print(f'Loading {len(wav_files)} files from {wav_dir} ...')
    for fn in tqdm(wav_files):
        signal, fs = wavread(fn)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if fs != target_fs:
            signal = librosa.resample(y=signal, orig_sr=fs, target_sr=target_fs)
        signals.append(signal)
        file_names.append(os.path.basename(fn))
    return signals, file_names

def extract_features(signal, sr=48000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(
        y=signal, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=1024,
        hop_length=512
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def load_labels_from_csv(csv_path):
    """Wczytuje labele z pliku CSV i tworzy mapowanie"""
    df = pd.read_csv(csv_path, sep=';')
    label_map = {}
    for _, row in df.iterrows():
        name = row['Name']
        # Tworzymy unikalny label z kombinacji wszystkich kolumn
        label = f"{row['Shaft']}_{row['Gears']}_{row['Tires']}_{row['Voltage']}"
        label_map[name] = label
    return label_map

def get_label_from_filename(filename, label_map, is_normal=False):
    """Wyciąga label z nazwy pliku"""
    if is_normal:
        return "Normal_Normal_Normal_Normal"
    
    # Szukamy identyfikatora w nazwie pliku (np. ab01, ab02)
    match = re.search(r'(ab\d+)', filename, re.IGNORECASE)
    if match:
        identifier = match.group(1).lower()
        if identifier in label_map:
            return label_map[identifier]
    
    # Jeśli nie znaleziono, zwracamy None
    return None

# =========================
# Wczytywanie labeli z CSV
# =========================
print(f"Loading labels from {labels_csv_path}...")
label_map = load_labels_from_csv(labels_csv_path)
print(f"Loaded {len(label_map)} label mappings")

# =========================
# Wczytywanie danych
# =========================
S_normal, normal_files = load_wavs_from_dir(normal_dir, target_fs)
S_anomaly, anomaly_files = load_wavs_from_dir(anomaly_dir, target_fs)

print(f'Loaded {len(S_normal)} normal samples')
print(f'Loaded {len(S_anomaly)} anomalous samples')

# =========================
# Tworzenie zbioru cech z labelami
# =========================
X = []
Y_labels = []

print("Extracting MFCC features and assigning labels...")
for sig, filename in tqdm(zip(S_normal, normal_files), total=len(S_normal), desc="Normal samples"):
    X.append(extract_features(sig, sr=target_fs))
    label = get_label_from_filename(filename, label_map, is_normal=True)
    Y_labels.append(label)

for sig, filename in tqdm(zip(S_anomaly, anomaly_files), total=len(S_anomaly), desc="Anomaly samples"):
    X.append(extract_features(sig, sr=target_fs))
    label = get_label_from_filename(filename, label_map, is_normal=False)
    if label is None:
        print(f"Warning: Could not find label for file {filename}, skipping...")
        continue
    Y_labels.append(label)

# Usuwamy próbki bez labeli
if len(X) != len(Y_labels):
    # To nie powinno się zdarzyć, ale na wszelki wypadek
    min_len = min(len(X), len(Y_labels))
    X = X[:min_len]
    Y_labels = Y_labels[:min_len]

X = np.array(X)

# Konwersja labeli string na numeryczne
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y_labels)

print("Feature matrix shape:", X.shape)
print(f"Number of unique labels: {len(label_encoder.classes_)}")
print("Labels:", label_encoder.classes_)

# =========================
# Trenowanie modelu
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=43, stratify=Y
)

clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =========================
# Zapisanie modelu
# =========================
os.makedirs(models_dir, exist_ok=True)
joblib.dump(clf, os.path.join(models_dir, f"model_{subdataset}.joblib"))
joblib.dump(scaler, os.path.join(models_dir, f"scaler_{subdataset}.joblib"))
joblib.dump(label_encoder, os.path.join(models_dir, f"label_encoder_{subdataset}.joblib"))
print(f"\nModel zapisany w {models_dir}/")
