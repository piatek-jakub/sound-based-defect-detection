import os
import sys
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# =========================
# Klasa do zapisywania wyjścia konsoli do pliku
# =========================
class Tee:
    """Klasa przechwytująca stdout i zapisująca do pliku oraz konsoli"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # Zapewnia natychmiastowy zapis
        self.stdout.write(text)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()
        sys.stdout = self.stdout

# =========================
# Parametry - ZMIEŃ TUTAJ TYP OBIEKTU
# =========================
# Dostępne opcje: "ToyCar", "ToyConveyor", "ToyTrain"
subdataset = "ToyConveyor"  # <-- ZMIEŃ TUTAJ

dataset_base = "ToyADMOS dataset"
target_fs = 48000
models_dir = "models"
labels_csv_path = f"labels/{subdataset}_anomay_condition.csv"

normal_dir = os.path.join(dataset_base, subdataset, "case1", "NormalSound_IND")
anomaly_dir = os.path.join(dataset_base, subdataset, "case1", "AnomalousSound_IND")

# =========================
# Inicjalizacja zapisu wyjścia konsoli do pliku
# =========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"train_log_{subdataset}_{timestamp}.txt")
tee = Tee(log_file_path)
sys.stdout = tee
print(f"Log zapisu treningu: {log_file_path}")
print("="*60)

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
    
    label_map = {}
    for _, row in df.iterrows():
        name = row['Name']
        # Zapisujemy wszystkie atrybuty dynamicznie
        label_map[name] = {attr: row[attr] for attr in attribute_columns}
    
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

def remove_old_models(models_dir, subdataset):
    """Usuwa stare modele dla danego typu obiektu przed zapisaniem nowych"""
    model_files = [
        f"model_{subdataset}.joblib",
        f"scaler_{subdataset}.joblib",
        f"label_encoders_{subdataset}.joblib"
    ]
    
    # Usuwamy również stare raporty
    report_pattern = os.path.join(models_dir, f"report_*_*_{subdataset}.png")
    old_reports = glob.glob(report_pattern)
    
    for file in model_files + old_reports:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Usunięto stary plik: {file_path}")

def create_classification_report_visualization(y_true, y_pred, label_encoder, attribute_name, 
                                             models_dir, subdataset, split_name='test'):
    """Tworzy wizualizację raportu klasyfikacji z confusion matrix i tabelą metryk"""
    # Pobieramy metryki
    report = classification_report(y_true, y_pred, 
                                  target_names=label_encoder.classes_, 
                                  output_dict=True)
    
    # Tworzymy confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Tworzymy figure z dwoma subplotami
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax1, cbar_kws={'label': 'Liczba próbek'})
    ax1.set_xlabel('Przewidywane', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rzeczywiste', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confusion Matrix - {attribute_name} ({split_name})', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
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
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Raport klasyfikacji - {attribute_name} ({split_name.upper()})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Zapisujemy
    output_path = os.path.join(models_dir, f"report_{attribute_name}_{split_name}_{subdataset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

# =========================
# Główny kod - opakowany w try-finally dla bezpiecznego zamykania pliku
# =========================
try:
    # =========================
    # Wczytywanie labeli z CSV
    # =========================
    print(f"Loading labels from {labels_csv_path}...")
    label_map, attribute_columns = load_labels_from_csv(labels_csv_path)
    print(f"Loaded {len(label_map)} label mappings")
    print(f"Attribute columns detected: {attribute_columns}")

    # =========================
    # Wczytywanie danych
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

    # Konwersja labeli string na numeryczne dla każdego atrybutu
    label_encoders = {}
    Y_encoded_list = []

    for attr in attribute_columns:
        label_encoders[attr] = LabelEncoder()
        Y_encoded_list.append(label_encoders[attr].fit_transform(Y_attributes[attr]))
        print(f"{attr} classes: {label_encoders[attr].classes_}")

    # Tworzymy macierz Y dla multi-output
    Y = np.column_stack(Y_encoded_list)

    print("Feature matrix shape:", X.shape)
    print("Label matrix shape:", Y.shape)
    print(f"Number of attributes: {len(attribute_columns)}")

    # =========================
    # Trenowanie modelu (multi-output)
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=43
    )

    base_clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Raporty dla każdego atrybutu osobno
    print("\n" + "="*60)
    print("Generowanie raportów wizualnych...")

    for i, attr_name in enumerate(attribute_columns):
        print(f"\n=== Classification Report - {attr_name} ===")
        print(classification_report(
            y_test[:, i], 
            y_pred[:, i], 
            target_names=label_encoders[attr_name].classes_
        ))
        
        # Tworzenie wizualizacji
        report_path = create_classification_report_visualization(
            y_test[:, i], 
            y_pred[:, i], 
            label_encoders[attr_name],
            attr_name,
            models_dir,
            subdataset,
            split_name='test'
        )
        print(f"Raport wizualny zapisany: {report_path}")

    # =========================
    # Zapisanie modelu
    # =========================
    os.makedirs(models_dir, exist_ok=True)

    # Usuwamy stare modele przed zapisaniem nowych
    remove_old_models(models_dir, subdataset)

    joblib.dump(clf, os.path.join(models_dir, f"model_{subdataset}.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, f"scaler_{subdataset}.joblib"))
    joblib.dump(label_encoders, os.path.join(models_dir, f"label_encoders_{subdataset}.joblib"))
    print(f"\nModel zapisany w {models_dir}/")
    print("="*60)
    print(f"Log zapisany w: {log_file_path}")
    
except Exception as e:
    print(f"\n{'='*60}")
    print(f"BŁĄD: {str(e)}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
    
finally:
    # Przywracanie stdout i zamykanie pliku (zawsze, nawet w przypadku błędu)
    sys.stdout = tee.stdout
    tee.close()
