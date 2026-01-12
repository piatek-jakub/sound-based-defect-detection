import os
import glob
import wave
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Ścieżka do datasetu
dataset_base = "ToyADMOS dataset"
labels_dir = "labels"

# Słownik do przechowywania statystyk
# Struktura: {kategoria: {atrybut: {wartość: {'count': int, 'total_duration': float}}}}
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_duration': 0.0})))

# Dodatkowe statystyki dla EnvironmentalNoise
env_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0})

print("="*80)
print("Analiza niezbalansowania datasetu ToyADMOS")
print("="*80)
print()

# Funkcja do odczytu długości pliku audio z metadanych (bez wczytywania całego pliku)
def get_audio_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()  # liczba ramek
            sample_rate = wav_file.getframerate()  # częstotliwość próbkowania
            duration = frames / float(sample_rate)  # długość w sekundach
            return duration
    except Exception as e:
        # Jeśli wave nie zadziała, spróbuj alternatywnej metody
        try:
            # Alternatywa: użyj struct do odczytu nagłówka WAV
            import struct
            with open(file_path, 'rb') as f:
                # Odczyt nagłówka WAV (44 bajty standardowego nagłówka)
                riff = f.read(4)
                if riff != b'RIFF':
                    return 0.0
                f.read(4)  # rozmiar pliku
                wave_header = f.read(4)
                if wave_header != b'WAVE':
                    return 0.0
                fmt = f.read(4)
                if fmt != b'fmt ':
                    return 0.0
                f.read(4)  # rozmiar fmt chunk
                f.read(2)  # format audio
                channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # byte rate
                f.read(2)  # block align
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                
                # Szukaj data chunk
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        return 0.0
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    if chunk_id == b'data':
                        # Długość = rozmiar danych / (sample_rate * channels * bits_per_sample/8)
                        bytes_per_sample = channels * (bits_per_sample // 8)
                        duration = chunk_size / (sample_rate * bytes_per_sample)
                        return duration
                    else:
                        f.read(chunk_size)  # pomiń ten chunk
        except Exception as e2:
            return 0.0

# Funkcja do wczytywania labels z CSV
def load_labels_from_csv(csv_path):
    """Wczytuje labele z pliku CSV i tworzy mapowanie"""
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', engine='python')
    except TypeError:
        try:
            df = pd.read_csv(csv_path, sep=';', error_bad_lines=False, warn_bad_lines=True, engine='python')
        except TypeError:
            df = pd.read_csv(csv_path, sep=';', error_bad_lines=False, warn_bad_lines=True)
    
    attribute_columns = [col for col in df.columns if col != 'Name']
    label_map = {}
    for _, row in df.iterrows():
        name = row['Name']
        if pd.isna(name):
            continue
        label_map[name] = {attr: row[attr] for attr in attribute_columns if pd.notna(row.get(attr))}
    
    return label_map, attribute_columns

# Funkcja do wyciągnięcia klas z nazwy pliku
def get_classes_from_filename(filename, label_map, attribute_columns, is_normal=False):
    """Wyciąga klasy z nazwy pliku"""
    if is_normal:
        # Dla normalnych próbek wszystkie atrybuty są "Normal"
        return {attr: 'Normal' for attr in attribute_columns}
    
    # Szukamy identyfikatora w nazwie pliku (np. ab01, ab02)
    match = re.search(r'(ab\d+)', filename, re.IGNORECASE)
    if match:
        identifier = match.group(1).lower()
        if identifier in label_map:
            return label_map[identifier]
    
    return None

# Przeszukiwanie wszystkich kategorii (ToyCar, ToyTrain, ToyConveyor)
categories = ['ToyCar', 'ToyTrain', 'ToyConveyor']

for category in categories:
    category_path = os.path.join(dataset_base, category)
    if not os.path.exists(category_path):
        print(f"Ostrzeżenie: Kategoria {category} nie istnieje")
        continue
    
    # Wczytaj labels dla tej kategorii
    labels_csv_path = os.path.join(labels_dir, f"{category}_anomay_condition.csv")
    if not os.path.exists(labels_csv_path):
        print(f"Ostrzeżenie: Brak pliku labels dla {category}")
        continue
    
    label_map, attribute_columns = load_labels_from_csv(labels_csv_path)
    print(f"Przetwarzanie kategorii: {category} (wczytano {len(label_map)} mapowań labeli)")
    
    # Przeszukiwanie wszystkich plików .wav w kategorii
    wav_files = glob.glob(os.path.join(category_path, '**', '*.wav'), recursive=True)
    
    for wav_file in tqdm(wav_files, desc=f"  {category}", leave=False):
        # Wyodrębnienie typu z ścieżki
        relative_path = os.path.relpath(wav_file, category_path)
        path_parts = relative_path.split(os.sep)
        
        # Określenie typu pliku
        if len(path_parts) == 1:
            # Plik bezpośrednio w kategorii (np. EnvironmentalNoise_CNT)
            file_type = os.path.basename(os.path.dirname(wav_file))
        else:
            # Plik w podfolderze (np. case1/AnomalousSound_IND)
            file_type = path_parts[-2] if len(path_parts) > 1 else os.path.basename(os.path.dirname(wav_file))
        
        filename = os.path.basename(wav_file)
        duration = get_audio_duration(wav_file)
        
        # Obsługa różnych typów plików
        if file_type == 'EnvironmentalNoise_CNT':
            env_stats[category]['count'] += 1
            env_stats[category]['total_duration'] += duration
        elif file_type == 'NormalSound_IND' or file_type == 'NormalSound_CNT':
            # Dla normalnych próbek wszystkie atrybuty są "Normal"
            classes = get_classes_from_filename(filename, label_map, attribute_columns, is_normal=True)
            if classes:
                for attr, value in classes.items():
                    stats[category][attr][value]['count'] += 1
                    stats[category][attr][value]['total_duration'] += duration
        elif file_type == 'AnomalousSound_IND':
            # Dla anomalii wyciągamy klasy z nazwy pliku
            classes = get_classes_from_filename(filename, label_map, attribute_columns, is_normal=False)
            if classes:
                for attr, value in classes.items():
                    stats[category][attr][value]['count'] += 1
                    stats[category][attr][value]['total_duration'] += duration

print()
print("="*80)
print("Zakończono analizę")
print("="*80)
print()

# Przygotowanie danych do wyświetlenia
results = []

# Dodaj statystyki dla każdej klasy (kategoria + atrybut + wartość)
for category in sorted(stats.keys()):
    for attr in sorted(stats[category].keys()):
        for value in sorted(stats[category][attr].keys()):
            count = stats[category][attr][value]['count']
            total_duration = stats[category][attr][value]['total_duration']
            if count > 0:  # Tylko jeśli są próbki
                results.append({
                    'Kategoria': category,
                    'Atrybut': attr,
                    'Wartość': value,
                    'Ilość próbek': count,
                    'Łączna długość (s)': round(total_duration, 2),
                    'Łączna długość (min)': round(total_duration / 60, 2),
                    'Łączna długość (h)': round(total_duration / 3600, 2)
                })

# Dodaj EnvironmentalNoise
for category in sorted(env_stats.keys()):
    count = env_stats[category]['count']
    total_duration = env_stats[category]['total_duration']
    if count > 0:
        results.append({
            'Kategoria': category,
            'Atrybut': 'EnvironmentalNoise',
            'Wartość': 'CNT',
            'Ilość próbek': count,
            'Łączna długość (s)': round(total_duration, 2),
            'Łączna długość (min)': round(total_duration / 60, 2),
            'Łączna długość (h)': round(total_duration / 3600, 2)
        })

# Tworzenie tabelki w formacie tekstowym
output_file = "dataset_balance_analysis.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*120 + "\n")
    f.write("ANALIZA NIEZBALANSOWANIA DATASETU TOYADMOS\n")
    f.write("="*120 + "\n\n")
    
    # Statystyki globalne
    total_samples = sum(r['Ilość próbek'] for r in results)
    total_duration = sum(r['Łączna długość (s)'] for r in results)
    
    f.write(f"STATYSTYKI GLOBALNE:\n")
    f.write(f"  Łączna ilość próbek: {total_samples:,}\n")
    f.write(f"  Łączna długość nagrań: {total_duration:,.2f} s ({total_duration/60:.2f} min, {total_duration/3600:.2f} h)\n")
    f.write("\n" + "="*120 + "\n\n")
    
    # Tabelka szczegółowa
    f.write("SZCZEGÓŁOWE STATYSTYKI DLA KAŻDEJ KLASY:\n")
    f.write("="*120 + "\n")
    
    # Nagłówki tabelki
    headers = ['Kategoria', 'Atrybut', 'Wartość', 'Ilość próbek', 'Długość (s)', 'Długość (min)', 'Długość (h)']
    col_widths = [12, 25, 30, 15, 15, 15, 15]
    
    # Nagłówek
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    f.write(header_line + "\n")
    f.write("-" * len(header_line) + "\n")
    
    # Wiersze danych
    for result in results:
        row = [
            result['Kategoria'].ljust(col_widths[0]),
            result['Atrybut'].ljust(col_widths[1]),
            result['Wartość'].ljust(col_widths[2]),
            str(result['Ilość próbek']).rjust(col_widths[3]),
            f"{result['Łączna długość (s)']:.2f}".rjust(col_widths[4]),
            f"{result['Łączna długość (min)']:.2f}".rjust(col_widths[5]),
            f"{result['Łączna długość (h)']:.2f}".rjust(col_widths[6])
        ]
        f.write(" | ".join(row) + "\n")
    
    f.write("\n" + "="*120 + "\n\n")
    
    # Analiza niezbalansowania dla każdej kategorii i atrybutu
    f.write("ANALIZA NIEZBALANSOWANIA WEDŁUG KATEGORII I ATRYBUTÓW:\n")
    f.write("="*120 + "\n\n")
    
    for category in sorted(stats.keys()):
        f.write(f"Kategoria: {category}\n")
        f.write("-" * 120 + "\n")
        
        for attr in sorted(stats[category].keys()):
            f.write(f"\n  Atrybut: {attr}\n")
            
            attr_results = [r for r in results if r['Kategoria'] == category and r['Atrybut'] == attr]
            if not attr_results:
                continue
            
            # Znajdź maksymalne wartości dla normalizacji
            max_count = max(r['Ilość próbek'] for r in attr_results)
            max_duration = max(r['Łączna długość (s)'] for r in attr_results)
            
            for result in sorted(attr_results, key=lambda x: x['Wartość']):
                count_ratio = (result['Ilość próbek'] / max_count * 100) if max_count > 0 else 0
                duration_ratio = (result['Łączna długość (s)'] / max_duration * 100) if max_duration > 0 else 0
                
                f.write(f"    {result['Wartość']:30s} | Próbki: {result['Ilość próbek']:6,} ({count_ratio:5.1f}%) | ")
                f.write(f"Długość: {result['Łączna długość (s)']:10.2f} s ({duration_ratio:5.1f}%)\n")
        
        # Dodaj EnvironmentalNoise jeśli istnieje
        if category in env_stats and env_stats[category]['count'] > 0:
            f.write(f"\n  EnvironmentalNoise:\n")
            f.write(f"    CNT: {env_stats[category]['count']:,} próbek, ")
            f.write(f"{env_stats[category]['total_duration']:.2f} s ({env_stats[category]['total_duration']/60:.2f} min)\n")
        
        f.write("\n")

print(f"Wyniki zapisano do pliku: {output_file}")

# Wyświetlenie podsumowania w konsoli
print("\nPODSUMOWANIE:")
print("-" * 80)
for category in sorted(stats.keys()):
    print(f"\n{category}:")
    for attr in sorted(stats[category].keys()):
        print(f"  {attr}:")
        for value in sorted(stats[category][attr].keys()):
            count = stats[category][attr][value]['count']
            duration = stats[category][attr][value]['total_duration']
            if count > 0:
                print(f"    {value:30s} | Próbki: {count:6,} | Długość: {duration:10.2f} s ({duration/60:6.2f} min)")

# EnvironmentalNoise
if env_stats:
    print("\nEnvironmentalNoise:")
    for category in sorted(env_stats.keys()):
        count = env_stats[category]['count']
        duration = env_stats[category]['total_duration']
        if count > 0:
            print(f"  {category}: {count:,} próbek, {duration:.2f} s ({duration/60:.2f} min)")
