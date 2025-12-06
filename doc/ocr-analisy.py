import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuracion de rutas dinamicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
INPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "computer-vision", "ocr_benchmark_results.csv")
OUTPUT_PLOT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "computer-vision", "ocr_score_dist.png")

def analyze_ocr_results():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: No se encuentra el archivo {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    total_imgs = len(df)
    
    print("-" * 50)
    print(f"REPORTE DE ANALISIS OCR | Total Imagenes: {total_imgs}")
    print("-" * 50)

    # 1. Estadisticas Generales
    mean_score = df['similarity_score'].mean()
    median_score = df['similarity_score'].median()
    std_dev = df['similarity_score'].std()
    perfect_reads = df[df['similarity_score'] == 1.0].shape[0]

    print(f"Promedio de Similitud: {mean_score*100:.2f}%")
    print(f"Mediana de Similitud:  {median_score*100:.2f}%")
    print(f"Lecturas Perfectas:    {perfect_reads} ({perfect_reads/total_imgs*100:.1f}%)")
    print("-" * 50)

    # 2. Desglose por Categorias de Calidad
    # Definimos rangos: Malo(<50%), Regular(50-80%), Bueno(80-99%), Perfecto(100%)
    bins = [-0.1, 0.5, 0.8, 0.999, 1.0]
    labels = ['Malo (<50%)', 'Regular (50-80%)', 'Bueno (80-99%)', 'Perfecto (100%)']
    
    df['calidad'] = pd.cut(df['similarity_score'], bins=bins, labels=labels)
    conteo = df['calidad'].value_counts().sort_index()
    porcentajes = df['calidad'].value_counts(normalize=True).sort_index() * 100

    print("DISTRIBUCION DE CALIDAD:")
    for label in labels:
        count = conteo.get(label, 0)
        pct = porcentajes.get(label, 0)
        print(f"  {label.ljust(18)}: {count} imgs ({pct:.1f}%)")

    # 3. Analisis de Fallos (Top 5 peores)
    print("-" * 50)
    print("TOP 5 PEORES LECTURAS (Para depuracion):")
    peores = df.sort_values('similarity_score').head(5)
    
    for i, row in peores.iterrows():
        print(f"\nID: {row['id']} | Score: {row['similarity_score']:.4f}")
        print(f"  Real: '{row['real_text']}'")
        print(f"  OCR:  '{row['extracted_text']}'")

    # 4. Generar Grafica de Distribucion
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='similarity_score', bins=20, kde=True, color='skyblue')
    plt.title(f'Distribucion de Similitud OCR (Promedio: {mean_score:.2f})')
    plt.xlabel('Similitud (Levenshtein Ratio)')
    plt.ylabel('Cantidad de Imagenes')
    plt.axvline(mean_score, color='red', linestyle='--', label=f'Promedio: {mean_score:.2f}')
    plt.legend()
    
    plt.savefig(OUTPUT_PLOT_PATH)
    print("-" * 50)
    print(f"Grafica de distribucion guardada en: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    analyze_ocr_results()