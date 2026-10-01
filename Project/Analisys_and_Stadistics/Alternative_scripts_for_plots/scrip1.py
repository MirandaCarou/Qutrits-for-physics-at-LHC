import ijson
import numpy as np
from decimal import Decimal
import time
import warnings
from IPython.display import clear_output
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from scipy.linalg import expm
import json
from scipy.stats import gaussian_kde
from scipy.linalg import qr
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
import itertools  # Importante para generar las combinaciones


def cargar_datos_json(json_path, num_jets=10000, num_constituents=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    eventos = []
    for i, evento in enumerate(data[:num_jets]):
        jet_pt = evento.get('jet_pt', i)
        jet_eta = evento.get('jet_eta', i)
        jet_phi = evento.get('jet_phi', i)
        jet_mass = evento.get('jet_sdmass', i)
        jet_energy = evento.get('jet_energy', i)
        jet_tau1 = evento.get('jet_tau1', i)
        jet_tau2 = evento.get('jet_tau2', i)
        jet_tau3 = evento.get('jet_tau3', i)    
        jet_tau4 = evento.get('jet_tau4', i)

        jet_tau12 = jet_tau2 / jet_tau1 if jet_tau1 != 0 else 0
        jet_tau23 = jet_tau3 / jet_tau2 if jet_tau2 != 0 else 0
        jet_tau34 = jet_tau4 / jet_tau3 if jet_tau3 != 0 else 0

        part_px = np.array(evento.get('part_px', []))
        part_py = np.array(evento.get('part_py', []))
        part_pz = np.array(evento.get('part_pz', []))

        part_energy = np.array(evento.get('part_energy', []))
        part_d0val = np.array(evento.get('part_d0val', []))
        part_dzval = np.array(evento.get('part_dzval', []))
        
        pt = np.sqrt(part_px**2 + part_py**2)
        p_total = np.sqrt(part_px**2 + part_py**2 + part_pz**2)
        eta = 0.5 * np.log((p_total + part_pz) / (p_total - part_pz + 1e-8)) 
        phi = np.arctan2(part_py, part_px)
        mass = np.sqrt(np.maximum(0, part_energy**2 - (part_px**2 + part_py**2 + part_pz**2)))
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
                'd0': part_d0val[idx],
                'dz': part_dzval[idx]
            })
            
        eventos.append({
            'pt_jet': jet_pt,
            'eta_jet': jet_eta,
            'phi_jet': jet_phi,
            'mass_jet': jet_mass,
            'energy_jet': jet_energy,
            'tau1_jet': jet_tau1,
            'tau2_jet': jet_tau2,
            'tau3_jet': jet_tau3,
            'tau4_jet': jet_tau4,
            'tau12_jet': jet_tau12,
            'tau23_jet': jet_tau23,
            'tau34_jet': jet_tau34,
            'constituents': top_constituents
        })

    return eventos

print('Empezó la carga de datos')
datos_HToBB = cargar_datos_json('./json/HToBB_120_flat.json', num_jets=10000, num_constituents=10)
datos_TTBar = cargar_datos_json('./json/TTBar_120_flat.json', num_jets=10000, num_constituents=10)
datos_WToqq = cargar_datos_json('./json/WToQQ_120_flat.json', num_jets=10000, num_constituents=10)
datos_QCD_simu_1 = cargar_datos_json('./json/ZJetsToNuNu_120_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_2 = cargar_datos_json('./json/ZJetsToNuNu_121_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_3 = cargar_datos_json('./json/ZJetsToNuNu_122_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_4 = cargar_datos_json('./json/ZJetsToNuNu_123_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_5 = cargar_datos_json('./json/ZJetsToNuNu_124_flat.json', num_jets=22500, num_constituents=10)
datos_QCD_simu_full = datos_QCD_simu_1 + datos_QCD_simu_2 + datos_QCD_simu_3 + datos_QCD_simu_4 + datos_QCD_simu_5

datos = np.array(datos_QCD_simu_full)
print('Datos cargados')

num_particles = 4

datos_filtrados = [
    jet for jet in datos 
    if len(jet['constituents']) >= num_particles
]

print(f"Jets antes: {len(datos)}")
print(f"Jets después de filtrar: {len(datos_filtrados)}")

datos = np.array(datos_filtrados)

# Generadores de Gell-Mann y matrices auxiliares
Lambda = {
    1: torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.cdouble),
    2: torch.tensor([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=torch.cdouble),
    3: torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.cdouble),
    4: torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=torch.cdouble),
    5: torch.tensor([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=torch.cdouble),
    6: torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.cdouble),
    7: torch.tensor([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=torch.cdouble),
    8: (1/torch.sqrt(torch.tensor(3.0))) * torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=torch.cdouble),
    0: torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)
}

Sigma = {
    1: (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.cdouble),
    2: (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=torch.cdouble),
    3: torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=torch.cdouble)
}

def TSWAP_matrix():
    tswap = np.zeros((9, 9), dtype=complex)
    for i in range(3):
        for j in range(3):
            ket = np.zeros(9)
            bra = np.zeros(9)
            ket[3*i + j] = 1   # |i⟩|j⟩
            bra[3*j + i] = 1   # |j⟩|i⟩
            tswap += np.outer(bra, ket)
    return tswap

def unitary_from_generator(generator_matrix, theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.cdouble)
    i = torch.tensor(1j, dtype=torch.cdouble)
    return Lambda[0] + (torch.cos(theta) - torch.tensor(1.0)) * generator_matrix @ generator_matrix + i * torch.sin(theta) * generator_matrix

def inicializing_qutrit_state(theta1, theta2, phi1, phi2):
    Gamma = torch.sqrt(torch.tensor(2.0)) * (torch.tensor(3.0) + torch.cos(theta1)*torch.cos(theta2) + torch.sin(theta1)*torch.sin(theta2)*torch.cos(phi1 - phi2))**(torch.tensor(-0.5))

    a0 = (torch.sqrt(torch.tensor(2.0)) * torch.cos(theta1/2) * torch.cos(theta2/2)).item()
    a1 = (torch.exp(1j * phi1) * torch.sin(theta1/2) * torch.cos(theta2/2) + torch.cos(theta1/2) * torch.sin(theta2/2) * torch.exp(1j * phi2)).item()
    a2 = (torch.sqrt(torch.tensor(2.0)) * torch.exp(1j * (phi1 + phi2)) * torch.sin(theta1/2) * torch.sin(theta2/2)).item()

    state = Gamma * torch.tensor([a0, a1, a2], dtype=torch.cdouble)
    state = state / torch.linalg.norm(state)
    return state.detach().clone().numpy()

def unitary_from_state(psi):
    psi = psi / np.linalg.norm(psi)
    a1 = torch.tensor([0.555, 0, 0.555], dtype=torch.cdouble)
    a2 = torch.tensor([0.555, 0.555, 0], dtype=torch.cdouble)
    mat = np.column_stack([psi, a1, a2])

    Q, R = qr(mat)
    phase = np.vdot(psi, Q[:, 0])
    Q[:, 0] = Q[:, 0] * (phase / abs(phase)).conj()
    return Q

# Parameters
num_particles = 4
num_latent = 1
num_ref = num_particles - num_latent
num_trash = num_ref
wires = list(range(num_particles + num_ref + 1))
ancilla = wires[-1]
dev = qml.device("default.qutrit", wires=wires)  

trash_wires = wires[num_latent : num_particles]
ref_wires = wires[num_particles : num_particles + num_trash]

# Encoding circuits base
def f(w): return 1 + (2 * np.pi / (1 + torch.exp(-w)))

def compute_feature(feature_name, w, constituent, jet):
    """Calcula el valor transformado según el nombre de la variable."""
    pt = constituent['pt']
    pt_jet = jet['pt_jet']
    factor = f(w) * (pt / pt_jet)
    
    if feature_name == 'eta':
        return factor * (constituent['eta'] - jet['eta_jet'])
    elif feature_name == 'phi':
        return factor * (constituent['phi'] - jet['phi_jet'])
    elif feature_name == 'energy':
        return factor * (constituent['energy'] - jet['energy_jet'])
    elif feature_name == 'mass':
        return factor * (constituent['mass'] - jet['mass_jet'])
    elif feature_name == 'tau12':
        return factor * jet['tau12_jet']
    elif feature_name == 'tau23':
        return factor * jet['tau23_jet']
    elif feature_name == 'tau34':
        return factor * jet['tau34_jet']
    elif feature_name == 'd0':
        return factor * constituent['d0']
    elif feature_name == 'dz':
        return factor * constituent['dz']
    else:
        raise ValueError(f"Variable de encoding no reconocida: {feature_name}")


# Dynamic Encoding Function
def encode_1p1q_qutrit(jets, w, unitaries, feature_combo):
    constituents = jets['constituents']
    v1, v2, v3, v4 = feature_combo  # Las 4 características a usar como [theta1, theta2, phi1, phi2]
        
    for i in range(num_particles):
        c = constituents[i]
        val1 = compute_feature(v1, w, c, jets)
        val2 = compute_feature(v2, w, c, jets)
        val3 = compute_feature(v3, w, c, jets)
        val4 = compute_feature(v4, w, c, jets)

        initial_state = inicializing_qutrit_state(val1, val2, val3, val4)
        u = unitary_from_state(initial_state)
        unitaries.append(u)
        qml.QutritUnitary(u, wires=i)    


def variational_layer_qutrit(theta_i, phi_i, w_i, num_layers):
    for layer in range(num_layers):
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                qml.TAdd(wires=[i, j])
        for i in range(num_particles):
            RX = unitary_from_generator(Sigma[1], phi_i[layer, i])
            RY = unitary_from_generator(Sigma[2], theta_i[layer, i])
            RZ = unitary_from_generator(Sigma[3], w_i[layer, i])
    
            qml.QutritUnitary(RX, wires=i)
            qml.QutritUnitary(RZ, wires=i)
            qml.QutritUnitary(RY, wires=i)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qae_circuit_qutrit(jets, w, theta_i, phi_i, w_i, num_layers, feature_combo):
    unitaries = []
    encode_1p1q_qutrit(jets, w, unitaries, feature_combo)
    variational_layer_qutrit(theta_i, phi_i, w_i, num_layers)
    tswap = TSWAP_matrix()

    for trash_wire, ref_wire in zip(trash_wires, ref_wires):
        qml.THadamard(wires=ancilla, subspace=None)
        qml.ControlledQutritUnitary(tswap, control_wires=ancilla, wires=[trash_wire, ref_wire])
        qml.THadamard(wires=ancilla, subspace=None)
    
    return qml.probs(wires=ancilla)

def cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, feature_combo):
    prob_0 = qae_circuit_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, feature_combo)[0]
    fidelity = prob_0
    return -fidelity, fidelity.item()

features_disponibles = ['eta', 'phi', 'energy', 'mass', 'tau12', 'tau23', 'tau34', 'd0', 'dz']
combinaciones = list(itertools.combinations(features_disponibles, 4))

print(f"Total de combinaciones a probar: {len(combinaciones)}")

fil_comb_back = []
fil_comb_HToBB = []
fil_comb_WToQQ = []
fil_comb_TTBar = []

# Iteración sobre cada combinación única de parámetros
for idx_comb, combo in enumerate(combinaciones):

    print(f"\n----- Inicio Combinación {idx_comb+1}/{len(combinaciones)}: {combo} --------")

    # Separación fija para consistencia en la comparación entre combinaciones
    X_train, X_temp = train_test_split(datos, train_size=10000, random_state=42, shuffle=True)
    X_val, rest = train_test_split(X_temp, train_size=2500, random_state=42, shuffle=True)
    X_inf, rest = train_test_split(rest, train_size=10000, random_state=42, shuffle=True)

    w = torch.tensor(1.0, requires_grad=True)
    num_layers = 1
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
    epoch_fidelities = []

    # Entrenamiento
    for epoch in range(num_epochs):
        for jet in X_train:
            if len(jet['constituents']) < num_particles:
                continue
        
            loss, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, combo)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_fidelities.append(fidelity)

    print("Fin de entrenamiento, empezamos inferencia...")
    event_fidelities_back = []
    event_fidelities_HToBB = []
    event_fidelities_WToQQ = []
    event_fidelities_TTBar = []

    for jet in X_inf:
        if len(jet['constituents']) < num_particles: continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, combo)
        event_fidelities_back.append(fidelity * 100) 

    for jet in datos_HToBB:
        if len(jet['constituents']) < num_particles: continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, combo)
        event_fidelities_HToBB.append(fidelity * 100) 

    for jet in datos_TTBar:
        if len(jet['constituents']) < num_particles: continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, combo)
        event_fidelities_TTBar.append(fidelity * 100) 

    for jet in datos_WToqq:
        if len(jet['constituents']) < num_particles: continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers, combo)
        event_fidelities_WToQQ.append(fidelity * 100) 

    fil_comb_back.append(event_fidelities_back)
    fil_comb_HToBB.append(event_fidelities_HToBB)
    fil_comb_WToQQ.append(event_fidelities_WToQQ)
    fil_comb_TTBar.append(event_fidelities_TTBar)

    print(f"------ Fin Combinación {idx_comb+1} -----------")

np.savez(
    './npz/np_qutrits_grid_search_encoding.npz',
    fil_back=fil_comb_back,
    fil_HToBB=fil_comb_HToBB,
    fil_WToQQ=fil_comb_WToQQ,
    fil_TTBar=fil_comb_TTBar,
    combinaciones=np.array(combinaciones, dtype=object)
)
print("Guardado exitosamente en ./npz/np_qutrits_grid_search_encoding.npz")