import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score
import sys

# -----------------------------------------------------------------------------
# 1. TRUCO DE COMPATIBILIDAD NUMPY / PYTHON 3.7
# -----------------------------------------------------------------------------
import numpy
sys.modules['numpy._core'] = numpy

# -----------------------------------------------------------------------------
# 2. FUNCIÓN PARA FORMATEAR COMBINACIONES EN LATEX
# -----------------------------------------------------------------------------
def format_combo_latex(combo_str):
    """
    Convierte una cadena de variables como 'eta, phi, tau12, energy'
    en una representación matemática en LaTeX: '$\\eta, \\phi, \\tau_{12}, E$'
    """
    mapping = {
        'eta': r'\eta',
        'phi': r'\phi',
        'energy': r'E',
        'mass': r'm',
        'tau12': r'\tau_{12}',
        'tau23': r'\tau_{23}',
        'tau34': r'\tau_{34}'
    }
    
    # Extraer las variables separadas por comas o espacios
    vars_list = [v.strip() for v in combo_str.replace('|', ',').split(',')]
    latex_vars = [mapping.get(v, v) for v in vars_list]
    
    return rf"${', '.join(latex_vars)}$"

# -----------------------------------------------------------------------------
# 3. CARGA DE DATOS Y SELECCIÓN DE COMBINACIONES (HToBB)
# -----------------------------------------------------------------------------
csv_path = './npz/metricas_detalladas_combinaciones.csv'
npz_path = './npz/np_qutrits_grid_search_encoding.npz'

df_metricas = pd.read_csv(csv_path)
data = np.load(npz_path, allow_pickle=True)

fil_back = data['fil_back']
fil_HToBB = data['fil_HToBB']

# Filtrar para HToBB y ordenar por AUC-ROC
df_htoBB = df_metricas[df_metricas['Proceso_Senal'] == 'HToBB'].copy()
df_htoBB = df_htoBB.sort_values(by='AUC_ROC', ascending=False).reset_index(drop=True)

# Selección de índices
all_ids = df_htoBB['ID_Combo'].tolist()
top_5_ids = df_htoBB.head(5)['ID_Combo'].tolist()
bottom_5_ids = df_htoBB.tail(5)['ID_Combo'].tolist()

# Lista de configuraciones para cada subplot
panel_configs = [
    {
        "title": "ROC Curves for all 35 encoding combinations",
        "ids": all_ids,
        "is_full": True
    },
    {
        "title": "ROC Curves for top 5 best performing encodings",
        "ids": top_5_ids,
        "is_full": False
    },
    {
        "title": "ROC Curves for bottom 5 best performing encodings",
        "ids": bottom_5_ids,
        "is_full": False
    }
]

# -----------------------------------------------------------------------------
# 4. PLOT EN GRIDSPEC (1 FILA x 3 COLUMNAS)
# -----------------------------------------------------------------------------
letters = "abcdefghijklmnopqrstuvwxyz"

fig = plt.figure(figsize=(22, 6.5))

gs = gridspec.GridSpec(
    1, 3,
    figure=fig,
    wspace=0.25
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
axes = [ax1, ax2, ax3]

for p_idx, config in enumerate(panel_configs):
    ax = axes[p_idx]
    combo_indices = config["ids"]
    is_full = config["is_full"]
    
    # Paleta de colores
    if is_full:
        colors = plt.cm.tab20(np.linspace(0, 1, len(combo_indices)))
    else:
        colors = plt.cm.tab10.colors

    for idx_rank, combo_id in enumerate(combo_indices):
        #Scores de anomalía (1 - fidelidad)
        scores_b = 1.0 - (np.array(fil_back[combo_id]) / 100.0)
        scores_s = 1.0 - (np.array(fil_HToBB[combo_id]) / 100.0)
        
        y_true = np.concatenate([np.zeros_like(scores_b), np.ones_like(scores_s)])
        y_scores = np.concatenate([scores_b, scores_s])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)
        
        raw_combo_name = df_htoBB[df_htoBB['ID_Combo'] == combo_id]['Combinacion'].values[0]
        latex_combo_name = format_combo_latex(raw_combo_name)
        
        # Formato de etiqueta
        if is_full:
            label = rf"C{combo_id+1} (AUC={auc_val:.3f})"
        else:
            label = rf"C{combo_id+1}: {latex_combo_name} (AUC={auc_val:.3f})"
            
        color = colors[idx_rank % len(colors)]
        
        ax.plot(
            fpr, tpr,
            lw=1.2 if is_full else 1.8,
            alpha=0.6 if is_full else 0.9,
            color=color,
            label=label
        )

    # Línea de referencia aleatoria
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        lw=1
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid(
        True,
        linestyle="-",
        color="whitesmoke",
        alpha=0.8
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=13
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=13
    )

    ax.set_title(
        f"({letters[p_idx]}) {config['title']}",
        fontsize=13,
        fontweight="bold"
    )

    ax.legend(
        fontsize=7.5 if is_full else 9,
        loc="lower right",
        frameon=True
    )

plt.suptitle(
    r"ROC Curves Comparison — $H \to b\bar{b}$ Signal",
    fontsize=18,
    fontweight="bold",
    y=1.02
)

plt.savefig(
    "ROC_encoding_comparison.pdf",
    bbox_inches="tight"
)

plt.show()