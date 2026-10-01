import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS Y SELECCIÓN DE COMBINACIONES
# -----------------------------------------------------------------------------
csv_path = './npz/metricas_detalladas_combinaciones.csv'
npz_path = './npz/np_qutrits_grid_search_encoding.npz'

# Cargar métricas agregadas del CSV y datos raw del NPZ para construir las ROC exactas
df_metricas = pd.read_csv(csv_path)

# Cargar arreglos de fidelidades usando el fix para Python 3.7 / NumPy
import sys
import numpy
sys.modules['numpy._core'] = numpy

data = np.load(npz_path, allow_pickle=True)
fil_back = data['fil_back']
fil_HToBB = data['fil_HToBB']

# Filtrar df para el proceso HToBB
df_htoBB = df_metricas[df_metricas['Proceso_Senal'] == 'HToBB'].copy()

# Ordenar por AUC-ROC descendente para determinar mayor/menor eficiencia
df_htoBB = df_htoBB.sort_values(by='AUC_ROC', ascending=False).reset_index(drop=True)

top_5_ids = df_htoBB.head(5)['ID_Combo'].tolist()
bottom_5_ids = df_htoBB.tail(5)['ID_Combo'].tolist()

# -----------------------------------------------------------------------------
# CONFIGURACIÓN ESTÉTICA DE MATPLOTLIB (ESTILO PUBLICACIÓN)
# -----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_roc_curves(combo_indices, title, output_filename, show_legend=True, is_full=False):
    plt.figure(figsize=(9, 7), dpi=300)
    
    # Colores continuos si son muchas combinaciones, o una paleta nítida si son 5
    if is_full:
        colors = plt.cm.tab20(np.linspace(0, 1, len(combo_indices)))
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for rank, idx in enumerate(combo_indices):
        # Reconstruir scores de anomalía (1 - fidelidad)
        scores_b = 1.0 - (np.array(fil_back[idx]) / 100.0)
        scores_s = 1.0 - (np.array(fil_HToBB[idx]) / 100.0)
        
        y_true = np.concatenate([np.zeros_like(scores_b), np.ones_like(scores_s)])
        y_scores = np.concatenate([scores_b, scores_s])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)
        
        combo_name = df_htoBB[df_htoBB['ID_Combo'] == idx]['Combinacion'].values[0]
        
        # Formato de etiqueta
        label_str = f"C{idx+1} ({combo_name}): AUC = {auc_val:.4f}"
        
        plt.plot(
            fpr, tpr, 
            color=colors[rank % len(colors)], 
            linewidth=2.0 if not is_full else 1.2, 
            alpha=0.9 if not is_full else 0.6,
            label=label_str
        )

    # Línea de clasificador aleatorio
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    
    if show_legend:
        legend_fontsize = 8 if is_full else 9.5
        plt.legend(
            bbox_to_anchor=(1.04, 1), 
            loc="upper left", 
            fontsize=legend_fontsize, 
            frameon=True, 
            facecolor='white', 
            edgecolor='none'
        )
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------------------------------
# 2. GENERACIÓN DE LAS TRES GRÁFICAS ROC (EN INGLÉS)
# -----------------------------------------------------------------------------

# Gráfica 1: Todas las 35 combinaciones
all_ids = df_htoBB['ID_Combo'].tolist()
plot_roc_curves(
    combo_indices=all_ids, 
    title='ROC Curves — All 35 Encoding Combinations (HToBB Signal)', 
    output_filename='./npz/roc_all_35_combinations.png',
    is_full=True
)

# Gráfica 2: Top 5 mejores combinaciones
plot_roc_curves(
    combo_indices=top_5_ids, 
    title='ROC Curves — Top 5 Best Performing Encodings (HToBB Signal)', 
    output_filename='./npz/roc_top_5_combinations.png',
    is_full=False
)

# Gráfica 3: Bottom 5 peores combinaciones
plot_roc_curves(
    combo_indices=bottom_5_ids, 
    title='ROC Curves — Bottom 5 Lowest Performing Encodings (HToBB Signal)', 
    output_filename='./npz/roc_bottom_5_combinations.png',
    is_full=False
)

print("\nFinished! Three ROC curves saved into './npz/' successfully.")