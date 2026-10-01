import sys
import numpy as np

# -----------------------------------------------------------------------------
# 1. FIX DE BACKEND PARA ENTORNO NO GRÁFICO / CLUSTER (CESGA)
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # Evita el error Gdk-CRITICAL al no requerir pantalla ($DISPLAY)
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 2. FIX DE COMPATIBILIDAD PICKLE PARA PYTHON 3.7 / NUMPY EN EL CLUSTER
# -----------------------------------------------------------------------------
try:
    import numpy._core.multiarray
except ImportError:
    # Mapea dinámicamente las llamadas modernas de numpy._core a numpy.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

from sklearn.metrics import roc_curve, roc_auc_score

# -----------------------------------------------------------------------------
# 3. CARGA DEL NPZ Y RESTO DEL SCRIPT
# -----------------------------------------------------------------------------
npz_path = './npz/np_qutrits_grid_search_encoding_holdout2.npz'
data = np.load(npz_path, allow_pickle=True)

fil_back = data['fil_back']
combinaciones = data['combinaciones']

def format_combo_latex(combo_tuple):
    """
    Convierte una tupla de variables (ej. ('eta', 'phi', 'd0', 'dz'))
    en representación matemática en LaTeX: '$\\eta, \\phi, d_0, d_z$'
    """
    mapping = {
        'eta': r'\eta',
        'phi': r'\phi',
        'energy': r'E',
        'mass': r'm',
        'tau12': r'\tau_{12}',
        'tau23': r'\tau_{23}',
        'tau34': r'\tau_{34}',
        'd0': r'd_0',
        'dz': r'd_z'
    }
    
    latex_vars = [mapping.get(v, v) for v in combo_tuple]
    return rf"${', '.join(latex_vars)}$"

# -----------------------------------------------------------------------------
# 3. CARGA EXCLUSIVA DEL NPZ
# -----------------------------------------------------------------------------
npz_path = './npz/np_qutrits_grid_search_encoding_holdout2.npz'
data = np.load(npz_path, allow_pickle=True)

fil_back = data['fil_back']
combinaciones = data['combinaciones']  # Tupla de 4 vars por combinación

# Configuración de los 3 procesos de señal
signals_config = {
    'HToBB': {
        'name': r'$H \to b\bar{b}$',
        'data': data['fil_HToBB'],
        'filename_prefix': 'htoBB'
    },
    'TTBar': {
        'name': r'$t\bar{t}$',
        'data': data['fil_TTBar'],
        'filename_prefix': 'ttbar'
    },
    'WToQQ': {
        'name': r'$W \to q\bar{q}$',
        'data': data['fil_WToQQ'],
        'filename_prefix': 'wtoqq'
    }
}

# -----------------------------------------------------------------------------
# 4. CONFIGURACIÓN ESTÉTICA DE PLOTTING
# -----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_roc_curves(combo_indices, fil_signal, combo_dict, title, output_filename, is_full=False):
    plt.figure(figsize=(9, 7), dpi=300)
    
    if is_full:
        colors = plt.cm.plasma(np.linspace(0, 1, len(combo_indices)))
    else:
        colors = plt.cm.tab10.colors

    for rank, idx in enumerate(combo_indices):
        # Reconstruir scores de anomalía (1 - fidelidad)
        scores_b = 1.0 - (np.array(fil_back[idx]) / 100.0)
        scores_s = 1.0 - (np.array(fil_signal[idx]) / 100.0)
        
        y_true = np.concatenate([np.zeros_like(scores_b), np.ones_like(scores_s)])
        y_scores = np.concatenate([scores_b, scores_s])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = combo_dict[idx]['auc']
        latex_name = format_combo_latex(combo_dict[idx]['combo_vars'])
        
        # Formato de etiqueta
        if is_full:
            label_str = rf"C{idx+1} (AUC = {auc_val:.4f})"
        else:
            label_str = rf"C{idx+1}: {latex_name} (AUC = {auc_val:.4f})"
        
        plt.plot(
            fpr, tpr, 
            color=colors[rank % len(colors)], 
            linewidth=0.8 if is_full else 2.0, 
            alpha=0.4 if is_full else 0.9,
            label=label_str if not is_full else None
        )

    # Línea de clasificador aleatorio
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    
    if not is_full:
        plt.legend(
            bbox_to_anchor=(1.04, 1), 
            loc="upper left", 
            fontsize=9.5, 
            frameon=True, 
            facecolor='white', 
            edgecolor='none'
        )
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Cierra la figura para liberar RAM en bucle

# -----------------------------------------------------------------------------
# 5. GENERACIÓN DE CURVAS ROC PARA CADA SEÑAL
# -----------------------------------------------------------------------------
for sig_key, sig_info in signals_config.items():
    print(f"\nProcesando señal: {sig_info['name']}...")
    fil_signal = sig_info['data']
    
    # 1. Calcular AUCs al vuelo para esta señal
    combo_performance = []
    for idx in range(len(combinaciones)):
        scores_b = 1.0 - (np.array(fil_back[idx]) / 100.0)
        scores_s = 1.0 - (np.array(fil_signal[idx]) / 100.0)
        
        y_true = np.concatenate([np.zeros_like(scores_b), np.ones_like(scores_s)])
        y_scores = np.concatenate([scores_b, scores_s])
        
        auc_val = roc_auc_score(y_true, y_scores)
        combo_performance.append({
            'ID_Combo': idx,
            'combo_vars': combinaciones[idx],
            'auc': auc_val
        })

    # 2. Ordenar de mayor a menor AUC
    combo_performance = sorted(combo_performance, key=lambda x: x['auc'], reverse=True)

    all_ids = [c['ID_Combo'] for c in combo_performance]
    top_5_ids = all_ids[:5]
    bottom_5_ids = all_ids[-5:]
    combo_dict = {c['ID_Combo']: c for c in combo_performance}

    prefix = sig_info['filename_prefix']
    sig_name = sig_info['name']

    # 3. Generar las 3 figuras para esta señal
    # Figura A: Todas las combinaciones
    plot_roc_curves(
        combo_indices=all_ids,
        fil_signal=fil_signal,
        combo_dict=combo_dict,
        title=rf'ROC Curves for all {len(all_ids)} encodings ({sig_name} Signal)',
        output_filename=f'./npz/roc_{prefix}_all_combinations.png',
        is_full=True
    )

    # Figura B: Top 5
    plot_roc_curves(
        combo_indices=top_5_ids,
        fil_signal=fil_signal,
        combo_dict=combo_dict,
        title=rf'ROC Curves for top 5 best encodings ({sig_name} Signal)',
        output_filename=f'./npz/roc_{prefix}_top_5.png',
        is_full=False
    )

    # Figura C: Bottom 5
    plot_roc_curves(
        combo_indices=bottom_5_ids,
        fil_signal=fil_signal,
        combo_dict=combo_dict,
        title=rf'ROC Curves for bottom 5 worst encodings ({sig_name} Signal)',
        output_filename=f'./npz/roc_{prefix}_bottom_5.png',
        is_full=False
    )

    print(f" Guardadas las 3 gráficas ROC para {sig_info['name']} en './npz/'")

print("\n¡Proceso global finalizado con éxito! Se han generado 9 figuras ROC en total.")