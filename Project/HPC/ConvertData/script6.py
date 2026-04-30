import h5py
import json
import numpy as np

def convert_h5_to_json_flat_pt(
        h5_file_path,
        json_file_path,
        max_events=50000,
        pt_min=500,
        pt_max=1000,
        nbins=10
    ):
    """
    Convierte un archivo HDF5 a JSON aplicando una selección que produce
    una distribución plana (flat) en jet_pt entre pt_min y pt_max.
    """

    try:
        with h5py.File(h5_file_path, 'r') as hf:

            jet_pts = hf['jet_kinematics'][:, 0]   # PT es el primer elemento
            total_events = len(jet_pts)

            # Seleccionar jets dentro del rango deseado
            mask = (jet_pts >= pt_min) & (jet_pts <= pt_max)
            pts_in_range = jet_pts[mask]
            idx_in_range = np.where(mask)[0]

            if len(idx_in_range) == 0:
                print("No hay jets entre ese rango de pt.")
                return

            # Crear bins
            bins = np.linspace(pt_min, pt_max, nbins + 1)

            # Obtener índice de bin de cada evento
            bin_indices = np.digitize(pts_in_range, bins) - 1

            # Asegurar que bins válidos
            valid = (bin_indices >= 0) & (bin_indices < nbins)
            pts_in_range = pts_in_range[valid]
            idx_in_range = idx_in_range[valid]
            bin_indices = bin_indices[valid]

            # Agrupar índices por bin
            bins_to_indices = {i: [] for i in range(nbins)}
            for idx, b in zip(idx_in_range, bin_indices):
                bins_to_indices[b].append(idx)

            # Número de jets a tomar por bin = mínimo entre los bins
            min_per_bin = min(len(v) for v in bins_to_indices.values())

            # Aplicar máximo total si se pide
            if min_per_bin * nbins > max_events:
                min_per_bin = max_events // nbins

            # Muestreo plano
            sampled_indices = []
            for b in range(nbins):
                indices = bins_to_indices[b]
                chosen = np.random.choice(indices, min_per_bin, replace=False)
                sampled_indices.extend(chosen)

            sampled_indices = sorted(sampled_indices)

            # Guardar a JSON
            data = []
            for i in sampled_indices:
                event = {
                    'event_info': hf['event_info'][i].tolist(),
                    'jet_kinematics': hf['jet_kinematics'][i].tolist(),
                    'PFCands': hf['PFCands'][i][np.any(hf['PFCands'][i] != 0, axis=1)].tolist(),
                    'jet_tagging': hf['jet_tagging'][i].tolist(),
                }
                data.append(event)

            with open(json_file_path, 'w') as jf:
                json.dump(data, jf, indent=4)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    h5_file = "RunG_batch0.h5"
    json_file = "runG_batch0_flatpt.json"
    convert_h5_to_json_flat_pt(h5_file, json_file)
