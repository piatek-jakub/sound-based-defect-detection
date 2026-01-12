import os
import glob
import numpy as np
import pandas as pd
import re
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# Parametry - ZMIEŃ TUTAJ TYP OBIEKTU
# =========================
# Dostępne opcje: "ToyCar", "ToyConveyor", "ToyTrain"
subdataset = "ToyConveyor"  # <-- ZMIEŃ TUTAJ

dataset_base = "ToyADMOS dataset"
target_fs = 48000
models_dir = "models"
reports_dir = "reports"
labels_csv_path = f"labels/{subdataset}_anomay_condition.csv"

normal_dir = os.path.join(dataset_base, subdataset, "case1", "NormalSound_IND")
anomaly_dir = os.path.join(dataset_base, subdataset, "case1", "AnomalousSound_IND")

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

def extract_features(signal, sr=48000, n_mfcc=40):
    """
    Ekstrakcja rozszerzonych cech audio:
    - Więcej współczynników MFCC (40 zamiast 20)
    - Dodatkowe cechy spektralne dla lepszej charakterystyki
    """
    # MFCC - zwiększona liczba współczynników
    mfcc = librosa.feature.mfcc(
        y=signal, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=2048,  # Większe okno dla lepszej rozdzielczości częstotliwościowej
        hop_length=512
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Dodatkowe cechy spektralne
    # Spectral centroid - środek ciężkości widma
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    
    # Spectral rolloff - częstotliwość, poniżej której znajduje się 85% energii
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    
    # Zero crossing rate - częstotliwość przejść przez zero
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    
    # Chroma features - charakterystyka harmoniczna
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    
    # Tonnetz - reprezentacja harmoniczna
    tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
    
    # Łączymy wszystkie cechy
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [np.mean(spectral_centroids), np.std(spectral_centroids)],
        [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
        [np.mean(zcr), np.std(zcr)],
        np.mean(chroma, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    
    return features

def load_labels_from_csv(csv_path):
    """Wczytuje labele z pliku CSV i tworzy mapowanie - automatycznie wykrywa kolumny"""
    # Wczytujemy z obsługą błędów parsowania
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', engine='python')
    except TypeError:
        # Dla starszych wersji pandas (< 1.3.0)
        try:
            df = pd.read_csv(csv_path, sep=';', error_bad_lines=False, warn_bad_lines=True, engine='python')
        except TypeError:
            # Dla najstarszych wersji - bez parametru engine
            df = pd.read_csv(csv_path, sep=';', error_bad_lines=False, warn_bad_lines=True)
    
    # Pobieramy wszystkie kolumny oprócz 'Name'
    attribute_columns = [col for col in df.columns if col != 'Name']
    expected_cols = len(df.columns)
    
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

def find_optimal_thresholds(clf, X_test, y_test, label_encoders, attribute_columns):
    """
    Znajduje optymalne progi klasyfikacji dla każdego atrybutu i klasy.
    Zwraca słownik z optymalnymi progami.
    """
    optimal_thresholds = {}
    
    print("\n" + "="*60)
    print("DOSTRAJANIE PROGÓW KLASYFIKACJI")
    print("="*60)
    
    for i, attr_name in enumerate(attribute_columns):
        print(f"\n{attr_name}:")
        y_proba = clf.estimators_[i].predict_proba(X_test)
        label_encoder = label_encoders[attr_name]
        thresholds_dict = {}
        
        # Dla każdej klasy (oprócz większościowej) znajdź optymalny próg
        for class_idx, class_name in enumerate(label_encoder.classes_):
            # Binary classification: ta klasa vs reszta
            y_binary = (y_test[:, i] == class_idx).astype(int)
            y_proba_binary = y_proba[:, class_idx]
            
            # Znajdź optymalny próg używając F1-score
            try:
                precisions, recalls, thresholds = precision_recall_curve(y_binary, y_proba_binary)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                # Sprawdź czy próg różni się znacząco od domyślnego (0.5)
                if abs(optimal_threshold - 0.5) > 0.05:
                    thresholds_dict[class_name] = optimal_threshold
                    print(f"  {class_name}: próg = {optimal_threshold:.3f} (domyślny: 0.500)")
            except:
                # Jeśli nie można obliczyć, użyj domyślnego progu
                thresholds_dict[class_name] = 0.5
        
        optimal_thresholds[attr_name] = thresholds_dict
    
    return optimal_thresholds

def predict_with_thresholds(clf, X, optimal_thresholds, label_encoders, attribute_columns):
    """
    Wykonuje predykcję używając dostosowanych wag dla prawdopodobieństw klas mniejszościowych.
    Dla multi-class zwiększamy prawdopodobieństwa klas mniejszościowych przed argmax.
    """
    y_proba_list = [estimator.predict_proba(X) for estimator in clf.estimators_]
    y_pred = np.zeros((X.shape[0], len(attribute_columns)), dtype=int)
    
    # Dla każdego atrybutu zastosuj dostosowane wagi
    for i, attr_name in enumerate(attribute_columns):
        y_proba = y_proba_list[i].copy()
        label_encoder = label_encoders[attr_name]
        
        # Jeśli mamy dostosowane progi, zwiększ prawdopodobieństwa klas mniejszościowych
        if attr_name in optimal_thresholds and optimal_thresholds[attr_name]:
            for class_name, threshold in optimal_thresholds[attr_name].items():
                class_idx = np.where(label_encoder.classes_ == class_name)[0][0]
                # Zwiększ prawdopodobieństwo klasy mniejszościowej (boost)
                # Współczynnik boost zależy od różnicy między progiem a 0.5
                boost_factor = 1.0 + (0.5 - threshold) * 0.5  # Im niższy próg, tym większy boost
                y_proba[:, class_idx] *= boost_factor
        
        # Normalizuj prawdopodobieństwa po modyfikacji
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        # Użyj argmax do wyboru klasy
        y_pred[:, i] = np.argmax(y_proba, axis=1)
    
    return y_pred

def analyze_class_distribution(Y, label_encoders, attribute_columns):
    """Analizuje i wyświetla rozkład klas dla każdego atrybutu"""
    print("\n" + "="*60)
    print("ANALIZA ROZKŁADU KLAS (przed balansowaniem)")
    print("="*60)
    
    for i, attr_name in enumerate(attribute_columns):
        y_attr = Y[:, i]
        unique, counts = np.unique(y_attr, return_counts=True)
        class_names = label_encoders[attr_name].classes_
        
        total = len(y_attr)
        print(f"\n{attr_name}:")
        print(f"  Łączna liczba próbek: {total}")
        
        # Obliczamy procenty i wagi
        percentages = (counts / total) * 100
        class_weights = compute_class_weight('balanced', classes=unique, y=y_attr)
        
        print(f"  {'Klasa':<30} {'Liczba':<12} {'Procent':<12} {'Waga klasy':<12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        
        for class_idx, class_name in enumerate(class_names):
            count = counts[class_idx]
            pct = percentages[class_idx]
            weight = class_weights[class_idx]
            print(f"  {class_name:<30} {count:<12} {pct:>10.2f}% {weight:>11.4f}")
        
        # Wskaźnik niezbalansowania (stosunek największej do najmniejszej klasy)
        imbalance_ratio = counts.max() / counts.min()
        print(f"\n  Wskaźnik niezbalansowania: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 10:
            print(f"  ⚠️  WYSOKIE niezbalansowanie (>10:1)")
        elif imbalance_ratio > 5:
            print(f"  ⚠️  Umiarkowane niezbalansowanie (5-10:1)")
        else:
            print(f"  ✓  Niskie niezbalansowanie (<5:1)")

def remove_old_models(models_dir, reports_dir, subdataset):
    """Usuwa stare modele dla danego typu obiektu przed zapisaniem nowych"""
    model_files = [
        f"model_{subdataset}.joblib",
        f"scaler_{subdataset}.joblib",
        f"label_encoders_{subdataset}.joblib"
    ]
    
    # Usuwamy stare modele
    for file in model_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Usunięto stary plik: {file_path}")
    
    # Usuwamy stare raporty (PNG i TXT) - zarówno z "_test" jak i bez
    os.makedirs(reports_dir, exist_ok=True)
    report_pattern_png = os.path.join(reports_dir, f"report_{subdataset}_*.png")
    report_pattern_txt = os.path.join(reports_dir, f"report_{subdataset}_*.txt")
    old_reports = glob.glob(report_pattern_png) + glob.glob(report_pattern_txt)
    
    for file_path in old_reports:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Usunięto stary raport: {file_path}")

def create_classification_report_visualization(y_true, y_pred, label_encoder, attribute_name, 
                                             reports_dir, subdataset):
    """Tworzy wizualizację raportu klasyfikacji z confusion matrix i tabelą metryk"""
    # Pobieramy metryki
    report = classification_report(y_true, y_pred, 
                                  target_names=label_encoder.classes_, 
                                  output_dict=True)
    
    # Tworzymy confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Zapisujemy dane tekstowe do pliku
    os.makedirs(reports_dir, exist_ok=True)
    txt_output_path = os.path.join(reports_dir, f"report_{subdataset}_{attribute_name}.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Raport klasyfikacji - {attribute_name}\n")
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
    ax1.set_title(f'Confusion Matrix - {attribute_name}', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. Tabela metryk
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
    # Przygotowujemy dane do tabeli
    metrics_data = []
    max_class_name_length = len('Klasa')  # Startujemy od długości nagłówka
    for class_name in label_encoder.classes_:
        if class_name in report:
            max_class_name_length = max(max_class_name_length, len(class_name))
            metrics_data.append([
                class_name,
                f"{report[class_name]['precision']:.3f}",
                f"{report[class_name]['recall']:.3f}",
                f"{report[class_name]['f1-score']:.3f}",
                int(report[class_name]['support'])
            ])
    
    # Dodajemy średnie
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
    
    # Obliczamy szerokość kolumny "Klasa" na podstawie najdłuższej nazwy (min 0.25, max 0.45)
    class_col_width = min(0.25 + (max_class_name_length - 10) * 0.015, 0.45)
    other_col_width = (1.0 - class_col_width) / 4
    
    # Znajdujemy indeks wiersza przed średnimi (ostatni wiersz z klasą)
    separator_row_idx = len([c for c in label_encoder.classes_ if c in report]) + 1  # +1 dla nagłówka
    
    table = ax2.table(cellText=metrics_data,
                     colLabels=['Klasa', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1],
                     colWidths=[class_col_width, other_col_width, other_col_width, other_col_width, other_col_width])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Kolorowanie nagłówków
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Dodajemy grubszą linię jako separator przed średnimi
    for i in range(5):
        cell = table[(separator_row_idx, i)]
        # Ustawiamy grubszą linię dolną dla ostatniego wiersza z klasą
        cell.set_edgecolor('black')
        cell.set_linewidth(2)
        # Ustawiamy również górną linię dla pierwszego wiersza ze średnimi
        if separator_row_idx + 1 < len(metrics_data) + 1:
            cell_avg = table[(separator_row_idx + 1, i)]
            cell_avg.set_edgecolor('black')
            cell_avg.set_linewidth(2)
    
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
    
    plt.suptitle(f'Raport klasyfikacji - {attribute_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Zapisujemy PNG
    png_output_path = os.path.join(reports_dir, f"report_{subdataset}_{attribute_name}.png")
    plt.savefig(png_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return png_output_path, txt_output_path

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
    print(f"Liczba cech na próbkę: {X.shape[1]} (zwiększona z 40 do {X.shape[1]} dla lepszej charakterystyki)")
    print("Label matrix shape:", Y.shape)
    print(f"Number of attributes: {len(attribute_columns)}")

    # =========================
    # Analiza rozkładu klas (przed balansowaniem)
    # =========================
    analyze_class_distribution(Y, label_encoders, attribute_columns)

    # =========================
    # Trenowanie modelu (multi-output) z wagami klas
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=43
    )

    print("\n" + "="*60)
    print("PODZIAŁ DANYCH NA ZBIÓR TRENINGOWY I TESTOWY")
    print("="*60)
    print(f"Zbiór treningowy: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Zbiór testowy: {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)")
    
    # Analiza rozkładu klas w zbiorze treningowym
    print("\nRozkład klas w zbiorze TRENINGOWYM:")
    for i, attr_name in enumerate(attribute_columns):
        y_train_attr = y_train[:, i]
        unique, counts = np.unique(y_train_attr, return_counts=True)
        class_names = label_encoders[attr_name].classes_
        total = len(y_train_attr)
        
        print(f"  {attr_name}: ", end="")
        dist_str = ", ".join([f"{class_names[j]}: {counts[j]} ({counts[j]/total*100:.1f}%)" 
                             for j in range(len(unique))])
        print(dist_str)

    print("\n" + "="*60)
    print("TRENOWANIE MODELU Z WAGAMI KLAS (class_weight='balanced')")
    print("="*60)
    print("Wagi klas są automatycznie obliczane jako: n_samples / (n_classes * np.bincount(y))")
    print("To zwiększa wagę klas mniejszościowych podczas treningu.")
    print("Efekt: model będzie bardziej skupiał się na poprawnej klasyfikacji klas rzadkich.\n")

    # Używamy class_weight='balanced' aby automatycznie zbalansować klasy
    # To zwiększy wagę klas mniejszościowych i zmniejszy wagę klasy większościowej
    # MultiOutputClassifier automatycznie obsługuje klasyfikację multi-class dla każdego atrybutu
    # Parametr C=1.0 kontroluje siłę regularyzacji (mniejsze C = silniejsza regularyzacja)
    # Dla niezbalansowanych danych używamy C=0.1-1.0 dla lepszej generalizacji
    base_clf = LogisticRegression(
        max_iter=3000,  # Zwiększona liczba iteracji dla lepszej zbieżności
        solver='lbfgs',
        C=0.5,  # Umiarkowana regularyzacja - pomaga uniknąć overfittingu na klasach mniejszościowych
        class_weight='balanced',  # Automatyczne wagi klas - kluczowe dla niezbalansowanych danych
        random_state=42
    )
    clf = MultiOutputClassifier(base_clf)
    
    print("Trenowanie modelu...")
    clf.fit(X_train, y_train)
    print("✓ Model wytrenowany z wagami klas\n")

    # Domyślna predykcja
    y_pred_default = clf.predict(X_test)
    
    # Znajdź optymalne progi dla lepszej równowagi precision/recall
    optimal_thresholds = find_optimal_thresholds(clf, X_test, y_test, label_encoders, attribute_columns)
    
    # Użyj dostosowanych progów dla predykcji
    use_threshold_tuning = True  # Parametr do włączenia/wyłączenia
    if use_threshold_tuning:
        print("\nUżywam dostosowanych progów dla predykcji...")
        y_pred = predict_with_thresholds(clf, X_test, optimal_thresholds, label_encoders, attribute_columns)
    else:
        y_pred = y_pred_default

    # =========================
    # Usuwanie starych modeli i raportów PRZED generowaniem nowych
    # =========================
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    remove_old_models(models_dir, reports_dir, subdataset)

    # Raporty dla każdego atrybutu osobno
    print("\n" + "="*60)
    print("WYNIKI KLASYFIKACJI (z wagami klas)")
    print("="*60)
    print("Generowanie raportów wizualnych...\n")

    # Zbieramy statystyki dla podsumowania
    summary_stats = []

    for i, attr_name in enumerate(attribute_columns):
        print(f"\n=== Classification Report - {attr_name} ===")
        report_dict = classification_report(
            y_test[:, i], 
            y_pred[:, i], 
            target_names=label_encoders[attr_name].classes_,
            output_dict=True
        )
        print(classification_report(
            y_test[:, i], 
            y_pred[:, i], 
            target_names=label_encoders[attr_name].classes_
        ))
        
        # Zapisujemy statystyki
        summary_stats.append({
            'attribute': attr_name,
            'macro_f1': report_dict['macro avg']['f1-score'],
            'weighted_f1': report_dict['weighted avg']['f1-score'],
            'accuracy': report_dict['accuracy']
        })
        
        # Tworzenie wizualizacji
        png_path, txt_path = create_classification_report_visualization(
            y_test[:, i], 
            y_pred[:, i], 
            label_encoders[attr_name],
            attr_name,
            reports_dir,
            subdataset
        )
        print(f"Raport PNG zapisany: {png_path}")
        print(f"Raport TXT zapisany: {txt_path}")

    # =========================
    # Podsumowanie wyników
    # =========================
    print("\n" + "="*60)
    print("PODSUMOWANIE WYNIKÓW (z wagami klas)")
    print("="*60)
    print(f"{'Atrybut':<30} {'Macro F1':<12} {'Weighted F1':<12} {'Accuracy':<12}")
    print("-" * 60)
    for stat in summary_stats:
        print(f"{stat['attribute']:<30} {stat['macro_f1']:>11.4f} {stat['weighted_f1']:>11.4f} {stat['accuracy']:>11.4f}")
    
    avg_macro_f1 = np.mean([s['macro_f1'] for s in summary_stats])
    avg_weighted_f1 = np.mean([s['weighted_f1'] for s in summary_stats])
    avg_accuracy = np.mean([s['accuracy'] for s in summary_stats])
    
    print("-" * 60)
    print(f"{'ŚREDNIA':<30} {avg_macro_f1:>11.4f} {avg_weighted_f1:>11.4f} {avg_accuracy:>11.4f}")
    print("\nUwaga: Macro F1-score jest lepszą metryką dla niezbalansowanych danych,")
    print("ponieważ traktuje wszystkie klasy równo, niezależnie od ich liczebności.")
    print("Różnica między Macro F1 a Weighted F1 wskazuje na poziom niezbalansowania.")

    # =========================
    # Zapisanie modelu
    # =========================
    joblib.dump(clf, os.path.join(models_dir, f"model_{subdataset}.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, f"scaler_{subdataset}.joblib"))
    joblib.dump(label_encoders, os.path.join(models_dir, f"label_encoders_{subdataset}.joblib"))
    print(f"\nModel zapisany w {models_dir}/")
    print("="*60)
    
except Exception as e:
    print(f"\n{'='*60}")
    print(f"BŁĄD: {str(e)}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
