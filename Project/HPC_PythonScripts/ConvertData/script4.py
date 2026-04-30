import h5py
import json
import numpy as np

def convert_h5_to_json(h5_file_path, json_file_path, max_events=22500):
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            data = []
            total_events = hf['event_info'].shape[0]
            N_jets = min(max_events, total_events)

            for i in range(N_jets):
                jet_kinematics = hf['jet_kinematics'][i]

                # Se asume que jet_pt está en la primera posición
                jet_pt = jet_kinematics[0]

                # Filtrado por jet_pt entre 500 y 1000 GeV (inclusive)
                if 500 <= jet_pt <= 1000:
                    event = {}

                    # Event Info
                    event['event_info'] = hf['event_info'][i].tolist()

                    # Jet Kinematics
                    event['jet_kinematics'] = jet_kinematics.tolist()

                    # PF Candidates
                    pfcands = hf['PFCands'][i]
                    valid_pfcands = pfcands[np.any(pfcands != 0, axis=1)].tolist()
                    event['PFCands'] = valid_pfcands

                    # Jet Tagging
                    event['jet_tagging'] = hf['jet_tagging'][i].tolist()

                    data.append(event)

            with open(json_file_path, 'w') as jf:
                json.dump(data, jf, indent=4)

            print(f"Se convirtieron {len(data)} eventos (jet_pt entre 500 y 1000 GeV) de '{h5_file_path}' a '{json_file_path}'.")

    except ImportError:
        print("Error: Asegúrate de tener instalados 'h5py' y 'json'.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{h5_file_path}'.")
    except KeyError as e:
        print(f"Error: No se encontró la clave '{e}' en el archivo HDF5. Verifica la estructura del archivo.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    h5_file = 'RunG_batch0.h5'         # Ruta del archivo .h5
    json_file = 'runG_batch0_jetpt500_1000.json'  # Ruta del archivo .json de salida
    convert_h5_to_json(h5_file, json_file, max_events=22500)
