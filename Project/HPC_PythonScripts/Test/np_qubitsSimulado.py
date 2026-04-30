import ijson
import numpy as np
import json
from decimal import Decimal
import json
import time
import torch
import warnings
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pennylane as qml
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


datos = []

def cargar_datos_json(json_path, num_jets=10000, num_constituents=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    eventos = []
    for i, evento in enumerate(data[:num_jets]):
        # Extract jet kinematics
        jet_pt = evento.get('jet_pt', i)
        jet_eta = evento.get('jet_eta', i)
        jet_phi = evento.get('jet_phi', i)
        jet_mass = evento.get('jet_sdmass', i)
        jet_energy = evento.get('jet_energy', i)


        # Extract constituents (particles)
        part_px = np.array(evento.get('part_px', []))
        part_py = np.array(evento.get('part_py', []))
        part_pz = np.array(evento.get('part_pz', []))
        part_energy = np.array(evento.get('part_energy', []))
        part_d0val = np.array(evento.get('part_d0val', []))
        part_dzval = np.array(evento.get('part_dzval', []))
        
        # Calculate pT, eta, phi, mass for each constituent
        pt = np.sqrt(part_px**2 + part_py**2)
        p_total = np.sqrt(part_px**2 + part_py**2 + part_pz**2)
        eta = 0.5 * np.log((p_total + part_pz) / (p_total - part_pz + 1e-8))  # Avoiding dividing by 0
        phi = np.arctan2(part_py, part_px)
        mass = np.sqrt(np.maximum(0, part_energy**2 - (part_px**2 + part_py**2 + part_pz**2)))
        
        # Seleccionar los num_constituents con mayor pt
        indices_ordenados = np.argsort(pt)[::-1][:num_constituents]
        
        top_constituents = []
        for idx in indices_ordenados:
            top_constituents.append({
                'pt': pt[idx],
                'eta': eta[idx],
                'phi': phi[idx],
                'px': part_px[idx],
                'py': part_py[idx],
                'pz': part_pz[idx],
                'mass': mass[idx],
                'energy': part_energy[idx],
            })
            
        eventos.append({
            'pt_jet': jet_pt,
            'eta_jet': jet_eta,
            'phi_jet': jet_phi,
            'mass_jet': jet_mass,
            'energy_jet': jet_energy,
            'constituents': top_constituents
        })
    print(f'Se cargaron {len(eventos)} eventos desde {json_path}')
    return eventos
# Ejemplo de uso
datos_HToBB_0 = cargar_datos_json('./json/HToBB_120_flat.json', num_jets=10000, num_constituents=10)
datos_TTBar_0 = cargar_datos_json('./json/TTBar_120_flat.json', num_jets=10000, num_constituents=10)
datos_WToqq_0 = cargar_datos_json('./json/WToQQ_120_flat.json', num_jets=10000, num_constituents=10)
datos_HToBB_1 = cargar_datos_json('./json/HToBB_121_flat.json', num_jets=10000, num_constituents=10)
datos_TTBar_1 = cargar_datos_json('./json/TTBar_121_flat.json', num_jets=10000, num_constituents=10)
datos_WToqq_1 = cargar_datos_json('./json/WToQQ_121_flat.json', num_jets=10000, num_constituents=10)
datos_QCD_simu_1 = cargar_datos_json('./json/ZJetsToNuNu_120_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_2 = cargar_datos_json('./json/ZJetsToNuNu_121_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_3 = cargar_datos_json('./json/ZJetsToNuNu_122_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_4 = cargar_datos_json('./json/ZJetsToNuNu_123_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_5 = cargar_datos_json('./json/ZJetsToNuNu_124_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_full = datos_QCD_simu_1 + datos_QCD_simu_2 + datos_QCD_simu_3 + datos_QCD_simu_4 + datos_QCD_simu_5
datos_HToBB = datos_HToBB_0 + datos_HToBB_1
datos_WToqq = datos_WToqq_0 + datos_WToqq_1
datos_TTBar = datos_TTBar_0 + datos_TTBar_1
datos = np.array(datos_QCD_simu_full)
print('Datos cargados')

num_particles = 10  # aseg      rate de que est       definido aqu

datos_filtrados = [
    jet for jet in datos 
    if len(jet['constituents']) >= num_particles
][:25000]

datos_filtradosH = [
    jet for jet in datos_HToBB 
    if len(jet['constituents']) >= num_particles
][:10000]

datos_filtradosW = [
    jet for jet in datos_WToqq 
    if len(jet['constituents']) >= num_particles
][:10000]

datos_filtradosT = [
    jet for jet in datos_TTBar 
    if len(jet['constituents']) >= num_particles
][:10000]

print("--- Verificación de longitudes ---")
print(f"Background (QCD): {len(datos_filtrados)} eventos")
print(f"HToBB:           {len(datos_filtradosH)} eventos")
print(f"WToQQ:           {len(datos_filtradosW)} eventos")
print(f"TTBar:           {len(datos_filtradosT)} eventos")

num_particles = 4
num_latent = 1
num_ref = num_particles - num_latent
num_trash = num_ref
wires = list(range(num_particles + num_ref + 1))  
ancilla = wires[-1]
dev = qml.device("default.qubit", wires=wires)

trash_wires = wires[num_latent : num_particles] # [1:4] si latent es 1 -> [1,2,3]
ref_wires = wires[num_particles : num_particles + num_trash]

def f(w):
    return 1 + (2 * np.pi / (1 + torch.exp(-w)))

def phi_circuit(w, phi, phi_jet, pt, pt_jet):
    return f(w) * pt / pt_jet * (phi - phi_jet)

def theta_circuit(w, eta, eta_jet, pt, pt_jet):
    return f(w) * pt / pt_jet * (eta - eta_jet)



def encode_1p1q(jet, w):
    pt_jet = jet['pt_jet']
    eta_jet = jet['eta_jet']
    phi_jet = jet['phi_jet']
    constituents = jet['constituents']

    for i in range(num_particles):
        c = constituents[i]
        theta = theta_circuit(w, c['eta'], eta_jet, c['pt'], pt_jet)
        phi = phi_circuit(w, c['phi'], phi_jet, c['pt'], pt_jet)
        qml.RY(theta, wires=i)
        qml.RX(phi, wires=i)


# --- Capa variacional ---
def variational_layer(theta_i, phi_i, w_i, num_layers):
    for layer in range(num_layers):
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                qml.CNOT(wires=[i, j])

        for i in range(num_particles):
            qml.RX(phi_i[layer, i], wires=i)
            qml.RZ(theta_i[layer, i], wires=i)
            qml.RY(w_i[layer, i], wires=i)



@qml.qnode(dev, interface="torch", diff_method="backprop")
def qae_circuit(jet, w, theta_i, phi_i, w_i, num_layers):
    encode_1p1q(jet, w)
    variational_layer(theta_i, phi_i, w_i, num_layers)

    for trash_wire, ref_wire in zip(trash_wires, ref_wires):
        qml.Hadamard(wires=ancilla)
        qml.CSWAP(wires=[ancilla, trash_wire, ref_wire])
        qml.Hadamard(wires=ancilla)

    return qml.probs(wires=ancilla)

def cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i, num_layers):
    prob_0 = qae_circuit(jet, w, theta_i, phi_i, w_i, num_layers)[0]
    fidelity = prob_0
    return -fidelity, fidelity.item()

def encontrar_maximos_per_jet(jet):
    max_pt = jet['pt_jet']
    max_eta = jet['eta_jet']
    max_phi = jet['phi_jet']
    return max_pt, max_eta, max_phi


# Hacer 100 veces 

fil_100_back = []
fil_100_HToBB = []
fil_100_WToQQ = []
fil_100_TTBar = []


for i in range(10):

    X_train, X_temp = train_test_split(
    	datos_filtrados, 
    	train_size=10000, 
    	random_state=i, 
    	shuffle=True
    )

    X_val, rest = train_test_split(
        X_temp, 
        train_size=2500, 
        random_state=i, 
        shuffle=True
    )

    X_inf, rest = train_test_split(
        rest, 
        train_size=10000, 
        random_state=i, 
        shuffle=True
    )


    print("------ Prueba número ", i+1, " ----------")
    w = torch.tensor(1.0, requires_grad=True)
    num_layers = 1 # Number of variational layers
    theta_i = (torch.rand(num_layers, num_particles) * 2 * torch.pi).requires_grad_(True)
    phi_i   = (torch.rand(num_layers, num_particles) * 2 * torch.pi).requires_grad_(True)
    w_i     = (torch.rand(num_layers, num_particles) * 2 * torch.pi).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [w, theta_i, phi_i, w_i],
        lr=5e-2,              
        betas=(0.5, 0.999),
        eps=1e-08,
        weight_decay=0.0,    
        amsgrad=True          
    )
    num_epochs = 1
    all_fidelities = []
    event_fidelities = [] 
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_fidelities = []
        avg_fidelity = 0.0
        avg_loss = 0.0
        
        for jet in X_train:
            if len(jet['constituents']) < num_particles:
                continue
        
            loss, fidelity = cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i,  num_layers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_fidelities.append(fidelity)
            event_fidelities.append(fidelity * 100)  # in %

        avg_loss = total_loss / len(epoch_fidelities)
        avg_fidelity = np.mean(epoch_fidelities) * 100
        all_fidelities.append(avg_fidelity)


    event_fidelities_back = []
    event_fidelities_HToBB = []
    event_fidelities_WToQQ = []
    event_fidelities_TTBar = []
    fidelidades = []
    etiquetas = []

    
    for jet in X_inf:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_back.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(0)

    

    inicio = time.time()
    for jet in datos_filtradosH:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_HToBB.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    for jet in datos_filtradosT:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_TTBar.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    for jet in datos_filtradosW:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_WToQQ.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    fil_100_back.append(event_fidelities_back)
    fil_100_HToBB.append(event_fidelities_HToBB)
    fil_100_WToQQ.append(event_fidelities_WToQQ)
    fil_100_TTBar.append(event_fidelities_TTBar)

    print("----Fin etapa ", i+1, "-----")




fil_100_back_qubits=fil_100_back
fil_100_HToBB_qubits=fil_100_HToBB
fil_100_WToQQ_qubits=fil_100_WToQQ
fil_100_TTBar_qubits=fil_100_TTBar

np.savez(
    './npz/np_qubitsSimulado.npz',
    fil_100_back_S=fil_100_back_qubits,
    fil_100_HToBB_S=fil_100_HToBB_qubits,
    fil_100_WToQQ_S=fil_100_WToQQ_qubits,
    fil_100_TTBar_S=fil_100_TTBar_qubits
)
print("Guardado como np_qubitsSimulado.npz")
