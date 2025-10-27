import h5py
import json
import numpy as np

def convert_h5_to_json(h5_file_path, json_file_path):
    """
    Converts an HDF5 file containing jet physics data to a JSON file.

    Args:
        h5_file_path (str): Path to the input .h5 file.
        json_file_path (str): Path to the output .json file.
    """
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            data = []
            N_jets = hf['event_info'].shape[0]

            for i in range(N_jets):
                event = {}

                # Event Info
                event['event_info'] = hf['event_info'][i].tolist()

                # Jet Kinematics
                event['jet_kinematics'] = hf['jet_kinematics'][i].tolist()

                # PF Candidates
                pfcands = hf['PFCands'][i]
                # Remove zero-padded candidates
                valid_pfcands = pfcands[np.any(pfcands != 0, axis=1)].tolist()
                event['PFCands'] = valid_pfcands

                # Jet Tagging
                event['jet_tagging'] = hf['jet_tagging'][i].tolist()

                data.append(event)

            with open(json_file_path, 'w') as jf:
                json.dump(data, jf, indent=4)  # Use indent for pretty formatting

            print(f"Successfully converted '{h5_file_path}' to '{json_file_path}'")

    except ImportError:
        print("Error: Please make sure you have 'h5py' and 'json' installed (json is usually built-in).")
    except FileNotFoundError:
        print(f"Error: The file '{h5_file_path}' was not found.")
    except KeyError as e:
        print(f"Error: Key '{e}' not found in the HDF5 file. Please check the file structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    h5_file = 'RunG_batch0.h5'  # Replace with the actual path to your .h5 file
    json_file = 'runG_batch0.json'      # Replace with the desired path for your .json file
    convert_h5_to_json(h5_file, json_file)
