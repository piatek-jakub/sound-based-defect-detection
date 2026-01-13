import os
import glob
import numpy as np
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# Parametry - ZMIEŃ TUTAJ TYP OBIEKTU
# =========================
# Dostępne opcje: "ToyCar", "ToyConveyor", "ToyTrain"
subdataset = "ToyTrain"  # <-- ZMIEŃ TUTAJ

dataset_base = "ToyADMOS dataset"
target_fs = 48000
models_dir = "models"
reports_dir = "reports"

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
    """Ekstrakcja cech audio"""
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=2048,
        hop_length=512
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Dodatkowe cechy spektralne
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
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

def analyze_class_distribution(y, label_encoder):
    """Analizuje i wyświetla rozkład klas"""
    print("\n" + "="*60)
    print("ANALIZA ROZKŁADU KLAS")
    print("="*60)
    
    unique, counts = np.unique(y, return_counts=True)
    class_names = label_encoder.classes_
    total = len(y)
    
    print(f"Łączna liczba próbek: {total}")
    print(f"\n{'Klasa':<30} {'Liczba':<12} {'Procent':<12} {'Waga klasy':<12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    
    class_weights = compute_class_weight('balanced', classes=unique, y=y)
    percentages = (counts / total) * 100
    
    for class_idx, class_name in enumerate(class_names):
        count = counts[class_idx]
        pct = percentages[class_idx]
        weight = class_weights[class_idx]
        print(f"{class_name:<30} {count:<12} {pct:>10.2f}% {weight:>11.4f}")
    
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nWskaźnik niezbalansowania: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 10:
        print(f"⚠️  WYSOKIE niezbalansowanie (>10:1)")
    elif imbalance_ratio > 5:
        print(f"⚠️  Umiarkowane niezbalansowanie (5-10:1)")
    else:
        print(f"✓  Niskie niezbalansowanie (<5:1)")

def remove_old_models(models_dir, reports_dir, subdataset):
    """Usuwa stare modele dla danego typu obiektu przed zapisaniem nowych"""
    model_files = [
        f"model_{subdataset}.joblib",
        f"scaler_{subdataset}.joblib",
        f"label_encoder_{subdataset}.joblib"
    ]
    
    for file in model_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Usunięto stary plik: {file_path}")
    
    os.makedirs(reports_dir, exist_ok=True)
    report_pattern_png = os.path.join(reports_dir, f"report_{subdataset}.png")
    report_pattern_txt = os.path.join(reports_dir, f"report_{subdataset}.txt")
    
    for file_path in [report_pattern_png, report_pattern_txt]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Usunięto stary raport: {file_path}")

def create_classification_report_visualization(y_true, y_pred, label_encoder, 
                                             reports_dir, subdataset):
    """Tworzy wizualizację raportu klasyfikacji"""
    report = classification_report(y_true, y_pred, 
                                  target_names=label_encoder.classes_, 
                                  output_dict=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Zapisujemy dane tekstowe do pliku
    os.makedirs(reports_dir, exist_ok=True)
    txt_output_path = os.path.join(reports_dir, f"report_{subdataset}.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Raport klasyfikacji - {subdataset}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
        f.write("\n\n")
        
        f.write("=== Confusion Matrix ===\n")
        f.write("Rzeczywiste \\ Przewidywane")
        for class_name in label_encoder.classes_:
            f.write(f"\t{class_name}")
        f.write("\n")
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_name}")
            for j in range(len(label_encoder.classes_)):
                f.write(f"\t{cm[i, j]}")
            f.write("\n")
        f.write("\n")
        
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
    ax1.set_title(f'Confusion Matrix - {subdataset}', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. Tabela metryk
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
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
    
    separator_row_idx = len([c for c in label_encoder.classes_ if c in report]) + 1
    
    table = ax2.table(cellText=metrics_data,
                     colLabels=['Klasa', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1],
                     colWidths=[0.3, 0.175, 0.175, 0.175, 0.175])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(5):
        cell = table[(separator_row_idx, i)]
        cell.set_edgecolor('black')
        cell.set_linewidth(2)
        if separator_row_idx + 1 < len(metrics_data) + 1:
            cell_avg = table[(separator_row_idx + 1, i)]
            cell_avg.set_edgecolor('black')
            cell_avg.set_linewidth(2)
    
    ax2.set_title(f'Metryki klasyfikacji - {subdataset}', 
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
    ax3.set_title(f'Metryki per klasa - {subdataset}', 
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=0, ha='center')
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Raport klasyfikacji - {subdataset}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    png_output_path = os.path.join(reports_dir, f"report_{subdataset}.png")
    plt.savefig(png_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return png_output_path, txt_output_path

# =========================
# Główny kod
# =========================
try:
    # =========================
    # Wczytywanie danych
    # =========================
    S_normal, normal_files = load_wavs_from_dir(normal_dir, target_fs)
    S_anomaly, anomaly_files = load_wavs_from_dir(anomaly_dir, target_fs)

    print(f'Loaded {len(S_normal)} normal samples')
    print(f'Loaded {len(S_anomaly)} anomalous samples')

    # =========================
    # Tworzenie zbioru cech z labelami (binarna klasyfikacja)
    # =========================
    X = []
    Y = []

    print("Extracting features and assigning labels...")
    for sig, filename in tqdm(zip(S_normal, normal_files), total=len(S_normal), desc="Normal samples"):
        X.append(extract_features(sig, sr=target_fs))
        Y.append('Normal')

    for sig, filename in tqdm(zip(S_anomaly, anomaly_files), total=len(S_anomaly), desc="Anomaly samples"):
        X.append(extract_features(sig, sr=target_fs))
        Y.append('Anomaly')

    X = np.array(X)
    
    # Konwersja labeli string na numeryczne
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    
    print(f"Klasy: {label_encoder.classes_}")

    print("Feature matrix shape:", X.shape)
    print(f"Liczba cech na próbkę: {X.shape[1]}")
    print("Label array shape:", Y_encoded.shape)

    # =========================
    # Analiza rozkładu klas
    # =========================
    analyze_class_distribution(Y_encoded, label_encoder)

    # =========================
    # Trenowanie modelu z wagami klas
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_encoded, test_size=0.2, random_state=43, stratify=Y_encoded
    )

    print("\n" + "="*60)
    print("PODZIAŁ DANYCH NA ZBIÓR TRENINGOWY I TESTOWY")
    print("="*60)
    print(f"Zbiór treningowy: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Zbiór testowy: {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\nRozkład klas w zbiorze TRENINGOWYM:")
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    for class_idx, class_name in enumerate(label_encoder.classes_):
        count = counts[class_idx]
        print(f"  {class_name}: {count} ({count/total*100:.1f}%)")

    print("\n" + "="*60)
    print("TRENOWANIE MODELU Z WAGAMI KLAS (class_weight='balanced')")
    print("="*60)
    print("Wagi klas są automatycznie obliczane jako: n_samples / (n_classes * np.bincount(y))")
    print("To zwiększa wagę klas mniejszościowych podczas treningu.\n")

    clf = LogisticRegression(
        max_iter=3000,
        solver='lbfgs',
        C=0.5,
        class_weight='balanced',
        random_state=42
    )
    
    print("Trenowanie modelu...")
    clf.fit(X_train, y_train)
    print("✓ Model wytrenowany z wagami klas\n")

    # =========================
    # Usuwanie starych modeli i raportów PRZED generowaniem nowych
    # =========================
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    remove_old_models(models_dir, reports_dir, subdataset)

    # =========================
    # Predykcja i raporty
    # =========================
    y_pred = clf.predict(X_test)

    print("\n" + "="*60)
    print("WYNIKI KLASYFIKACJI")
    print("="*60)
    print("Generowanie raportów wizualnych...\n")

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Tworzenie wizualizacji
    png_path, txt_path = create_classification_report_visualization(
        y_test, 
        y_pred, 
        label_encoder,
        reports_dir,
        subdataset
    )
    print(f"\nRaport PNG zapisany: {png_path}")
    print(f"Raport TXT zapisany: {txt_path}")

    # =========================
    # Podsumowanie wyników
    # =========================
    report_dict = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print("\n" + "="*60)
    print("PODSUMOWANIE WYNIKÓW")
    print("="*60)
    print(f"Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"Accuracy: {report_dict['accuracy']:.4f}")

    # =========================
    # Zapisanie modelu
    # =========================
    joblib.dump(clf, os.path.join(models_dir, f"model_{subdataset}.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, f"scaler_{subdataset}.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, f"label_encoder_{subdataset}.joblib"))
    print(f"\nModel zapisany w {models_dir}/")
    print("="*60)
    
except Exception as e:
    print(f"\n{'='*60}")
    print(f"BŁĄD: {str(e)}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
