import uproot
import json
import numpy as np

def convert_root_to_json_flat_pt(root_file_path, json_file_path, max_events=100000, pt_range=(500, 1000), n_bins=10):
    with uproot.open(root_file_path) as file:
        tree = file["tree"]
        branches = tree.keys()
        arrays = tree.arrays(branches, entry_stop=max_events)

        jet_pts = arrays["jet_pt"]
        mask = (jet_pts >= pt_range[0]) & (jet_pts <= pt_range[1])
        indices_in_range = np.where(mask)[0]

        if len(indices_in_range) == 0:
            print(f"No jets in range {pt_range} for {root_file_path}")
            return

        # Dividir el rango de pT en partes
        bins = np.linspace(pt_range[0], pt_range[1], n_bins + 1)
        digitized = np.digitize(jet_pts[indices_in_range], bins)

        # Calcular el número mínimo de eventos por parte
        counts = np.bincount(digitized, minlength=n_bins + 2)[1:-1]  # quitar bordes vacíos
        min_count = np.min(counts[counts > 0])

        # Seleccionar el mismo número de eventos por parte
        sampled_indices = []
        for b in range(1, n_bins + 1):
            bin_indices = indices_in_range[digitized == b]
            if len(bin_indices) >= min_count:
                sampled = np.random.choice(bin_indices, min_count, replace=False)
                sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        # Convertir los eventos seleccionados a formato JSON
        data = []
        for i in sampled_indices:
            event = {}
            for branch in branches:
                value = arrays[branch][i]
                if isinstance(value, np.ndarray):
                    event[branch] = value.tolist()
                elif hasattr(value, "tolist"):
                    event[branch] = value.tolist()
                else:
                    event[branch] = value
            data.append(event)

        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Archivo convertido exitosamente a '{json_file_path}' con distribución plana en jet pT [{pt_range[0]}, {pt_range[1]}] GeV")


# Uso del script
if __name__ == "__main__":
    convert_root_to_json_flat_pt("HToBB_120.root", "HToBB_120_flat.json")
    convert_root_to_json_flat_pt("TTBar_120.root", "TTBar_120_flat.json")
    convert_root_to_json_flat_pt("WToQQ_120.root", "WToQQ_120_flat.json")
    convert_root_to_json_flat_pt("ZJetsToNuNu_120.root", "ZJetsToNuNu_120_flat.json")
