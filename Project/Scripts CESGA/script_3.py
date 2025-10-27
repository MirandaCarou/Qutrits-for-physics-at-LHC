import uproot
import json
import numpy as np

def convert_root_to_json(root_file_path, json_file_path, max_events=10000):
    with uproot.open(root_file_path) as file:
        tree = file["tree"]  # Usamos "tree" según tu descripción
        branches = tree.keys()
        arrays = tree.arrays(branches, entry_stop=max_events)

        data = []
        num_events = len(arrays["jet_pt"])  # Usamos una rama escalar para contar eventos

        for i in range(num_events):
            event = {}
            for branch in branches:
                value = arrays[branch][i]
                # Convertimos listas de partículas o valores escalares
                if isinstance(value, np.ndarray):
                    event[branch] = value.tolist()
                elif hasattr(value, "tolist"):  # tipos NumPy escalares
                    event[branch] = value.tolist()
                else:
                    event[branch] = value
            data.append(event)

        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Archivo convertido exitosamente a '{json_file_path}'")

# Uso del script
if __name__ == "__main__":
    convert_root_to_json("TTBar_120.root", "TTBar_120.json", max_events=10000)
    convert_root_to_json("WToQQ_120.root", "WToQQ_120.json", max_events=10000)
