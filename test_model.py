import os
import glob
import numpy as np
import pandas as pd
import re
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# Parametry - ZMIEŃ TUTAJ TYP OBIEKTU
# =========================
# Dostępne opcje: "ToyCar", "ToyConveyor", "ToyTrain"
subdataset = "ToyCar"  # <-- ZMIEŃ TUTAJ

dataset_base = "ToyADMOS dataset"
target_fs = 48000
models_dir = "models"
reports_dir = "reports"
labels_csv_path = f"labels/{subdataset}_anomay_condition.csv"

model_path = os.path.join(models_dir, f"model_{subdataset}.joblib")
scaler_path = os.path.join(models_dir, f"scaler_{subdataset}.joblib")
label_encoders_path = os.path.join(models_dir, f"label_encoders_{subdataset}.joblib")

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
    """Wczytuje labele z pliku CSV i tworzy mapowanie - automatycznie wykrywa kolumny"""
    df = pd.read_csv(csv_path, sep=';')
    # Pobieramy wszystkie kolumny oprócz 'Name'
    attribute_columns = [col for col in df.columns if col != 'Name']
    
    # Walidacja - sprawdzamy czy wszystkie wiersze mają poprawną liczbę kolumn
    invalid_rows = []
    for idx, row in df.iterrows():
        # Sprawdzamy czy są wartości NaN (oznacza brakujące kolumny)
        if row.isna().any():
            invalid_rows.append((idx + 2, row['Name'] if pd.notna(row.get('Name')) else f'row {idx + 2}'))
    
    if invalid_rows:
        print(f"UWAGA: Znaleziono {len(invalid_rows)} wierszy z brakującymi wartościami w {csv_path}:")
        for row_num, name in invalid_rows:
            print(f"  - Linia {row_num}: {name}")
        print("Te wiersze mogą powodować problemy w klasyfikacji!")
    
    label_map = {}
    for _, row in df.iterrows():
        name = row['Name']
        # Pomijamy wiersze z brakującymi wartościami
        if pd.isna(name):
            continue
        # Zapisujemy wszystkie atrybuty dynamicznie
        label_map[name] = {attr: row[attr] for attr in attribute_columns if pd.notna(row.get(attr))}
    
    return label_map, attribute_columns

def get_label_from_filename(filename, label_map, attribute_columns, is_normal=False):
    """Wyciąga labele z nazwy pliku - zwraca krotkę wartości dla wszystkich atrybutów"""
    if is_normal:
        # Dla normalnych próbek wszystkie atrybuty są "Normal"
        return tuple(['Normal'] * len(attribute_columns))
    
    # Szukamy identyfikatora w nazwie pliku (np. ab01, ab02)
    match = re.search(r'(ab\d+)', filename, re.IGNORECASE)
    if match:
        identifier = match.group(1).lower()
        if identifier in label_map:
            labels = label_map[identifier]
            return tuple([labels[attr] for attr in attribute_columns])
    
    # Jeśli nie znaleziono, zwracamy None
    return None

def create_classification_report_visualization(y_true, y_pred, label_encoder, attribute_name, 
                                             reports_dir, subdataset, split_name='test'):
    """Tworzy wizualizację raportu klasyfikacji z confusion matrix i tabelą metryk"""
    # Pobieramy metryki
    report = classification_report(y_true, y_pred, 
                                  target_names=label_encoder.classes_, 
                                  output_dict=True)
    
    # Tworzymy confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Zapisujemy dane tekstowe do pliku
    os.makedirs(reports_dir, exist_ok=True)
    txt_output_path = os.path.join(reports_dir, f"report_{subdataset}_{attribute_name}_{split_name}.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Raport klasyfikacji - {attribute_name} ({split_name.upper()})\n")
        f.write(f"Dataset: {subdataset}\n")
        f.write(f"{'='*60}\n\n")
        
        # Classification Report
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
        f.write("\n\n")
        
        # Confusion Matrix
        f.write("=== Confusion Matrix ===\n")
        f.write(f"Klasa: {attribute_name}\n\n")
        # Nagłówek
        f.write("Rzeczywiste \\ Przewidywane")
        for class_name in label_encoder.classes_:
            f.write(f"\t{class_name}")
        f.write("\n")
        # Wiersze
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_name}")
            for j in range(len(label_encoder.classes_)):
                f.write(f"\t{cm[i, j]}")
            f.write("\n")
        f.write("\n")
        
        # Szczegółowe metryki
        f.write("=== Szczegółowe metryki ===\n")
        for class_name in label_encoder.classes_:
            if class_name in report:
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
                f.write(f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n")
                f.write(f"  Support:   {int(report[class_name]['support'])}\n")
        
        f.write(f"\n\nMacro Average:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}\n")
        
        f.write(f"\nWeighted Average:\n")
        f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
    
    # Tworzymy figure z dwoma subplotami
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax1, cbar_kws={'label': 'Liczba próbek'})
    ax1.set_xlabel('Przewidywane', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Rzeczywiste', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confusion Matrix - {attribute_name} ({split_name})', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. Tabela metryk
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
    # Przygotowujemy dane do tabeli
    metrics_data = []
    for class_name in label_encoder.classes_:
        if class_name in report:
            metrics_data.append([
                class_name,
                f"{report[class_name]['precision']:.3f}",
                f"{report[class_name]['recall']:.3f}",
                f"{report[class_name]['f1-score']:.3f}",
                int(report[class_name]['support'])
            ])
    
    # Dodajemy średnie
    metrics_data.append(['', '', '', '', ''])
    metrics_data.append([
        'Macro Avg',
        f"{report['macro avg']['precision']:.3f}",
        f"{report['macro avg']['recall']:.3f}",
        f"{report['macro avg']['f1-score']:.3f}",
        int(report['macro avg']['support'])
    ])
    metrics_data.append([
        'Weighted Avg',
        f"{report['weighted avg']['precision']:.3f}",
        f"{report['weighted avg']['recall']:.3f}",
        f"{report['weighted avg']['f1-score']:.3f}",
        int(report['weighted avg']['support'])
    ])
    
    table = ax2.table(cellText=metrics_data,
                     colLabels=['Klasa', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Kolorowanie nagłówków
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title(f'Metryki klasyfikacji - {attribute_name}', 
                  fontsize=12, fontweight='bold', pad=20)
    
    # 3. Wykres słupkowy metryk
    ax3 = fig.add_subplot(gs[1, 1])
    classes = [c for c in label_encoder.classes_ if c in report]
    precision_vals = [report[c]['precision'] for c in classes]
    recall_vals = [report[c]['recall'] for c in classes]
    f1_vals = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax3.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
    ax3.bar(x, recall_vals, width, label='Recall', alpha=0.8)
    ax3.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)
    
    ax3.set_xlabel('Klasa', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Wartość', fontsize=11, fontweight='bold')
    ax3.set_title(f'Metryki per klasa - {attribute_name}', 
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=0, ha='center')
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Raport klasyfikacji - {attribute_name} ({split_name.upper()})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Zapisujemy PNG
    png_output_path = os.path.join(reports_dir, f"report_{subdataset}_{attribute_name}_{split_name}.png")
    plt.savefig(png_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return png_output_path, txt_output_path

# =========================
# Wczytywanie modelu
# =========================
print(f"Loading model from {model_path}...")
clf = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoders_path)

# =========================
# Wczytywanie labeli z CSV
# =========================
print(f"Loading labels from {labels_csv_path}...")
label_map, attribute_columns = load_labels_from_csv(labels_csv_path)
print(f"Loaded {len(label_map)} label mappings")
print(f"Attribute columns detected: {attribute_columns}")

# =========================
# Wczytywanie danych testowych
# =========================
S_normal, normal_files = load_wavs_from_dir(normal_dir, target_fs)
S_anomaly, anomaly_files = load_wavs_from_dir(anomaly_dir, target_fs)

print(f'Loaded {len(S_normal)} normal samples')
print(f'Loaded {len(S_anomaly)} anomalous samples')

# =========================
# Tworzenie zbioru cech z labelami (multi-label)
# =========================
X = []
Y_labels = []

print("Extracting MFCC features and assigning labels...")
for sig, filename in tqdm(zip(S_normal, normal_files), total=len(S_normal), desc="Normal samples"):
    X.append(extract_features(sig, sr=target_fs))
    label = get_label_from_filename(filename, label_map, attribute_columns, is_normal=True)
    Y_labels.append(label)

for sig, filename in tqdm(zip(S_anomaly, anomaly_files), total=len(S_anomaly), desc="Anomaly samples"):
    X.append(extract_features(sig, sr=target_fs))
    label = get_label_from_filename(filename, label_map, attribute_columns, is_normal=False)
    if label is None:
        print(f"Warning: Could not find label for file {filename}, skipping...")
        continue
    Y_labels.append(label)

# Usuwamy próbki bez labeli
if len(X) != len(Y_labels):
    min_len = min(len(X), len(Y_labels))
    X = X[:min_len]
    Y_labels = Y_labels[:min_len]

X = np.array(X)

# Rozdzielamy labele na osobne kolumny dla każdego atrybutu
Y_attributes = {}
for i, attr in enumerate(attribute_columns):
    Y_attributes[attr] = [label[i] for label in Y_labels]

# Konwersja labeli string na numeryczne używając tych samych encoder'ów co podczas treningu
Y_encoded_list = []
for attr in attribute_columns:
    Y_encoded_list.append(label_encoders[attr].transform(Y_attributes[attr]))

# Tworzymy macierz Y dla multi-output
Y = np.column_stack(Y_encoded_list)

print("Feature matrix shape:", X.shape)
print("Label matrix shape:", Y.shape)
print(f"Number of attributes: {len(attribute_columns)}")

# =========================
# Testowanie modelu
# =========================
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)

# Raporty dla każdego atrybutu osobno
print("\n" + "="*60)
print("Generowanie raportów wizualnych...")

for i, attr_name in enumerate(attribute_columns):
    print(f"\n=== Classification Report - {attr_name} ===")
    print(classification_report(
        Y[:, i], 
        y_pred[:, i], 
        target_names=label_encoders[attr_name].classes_
    ))
    
    # Tworzenie wizualizacji
    png_path, txt_path = create_classification_report_visualization(
        Y[:, i], 
        y_pred[:, i], 
        label_encoders[attr_name],
        attr_name,
        reports_dir,
        subdataset,
        split_name='test'
    )
    print(f"Raport PNG zapisany: {png_path}")
    print(f"Raport TXT zapisany: {txt_path}")
