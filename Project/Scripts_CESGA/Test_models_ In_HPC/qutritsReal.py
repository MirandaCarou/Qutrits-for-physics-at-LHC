import ijson
import numpy as np
import json
from decimal import Decimal
import json
import time
import torch
import warnings
import numpy as np
from IPython.display import clear_output
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pennylane as qml
import numpy as np
import torch
from scipy.linalg import expm
import json
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from scipy.linalg import qr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np




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
        jet_tau1 = evento.get('jet_tau1', i)
        jet_tau2 = evento.get('jet_tau2', i)
        jet_tau3 = evento.get('jet_tau3', i)    
        jet_tau4 = evento.get('jet_tau4', i)

        jet_tau12 = jet_tau1 / jet_tau2 if jet_tau2 != 0 else 0
        jet_tau23 = jet_tau2 / jet_tau3 if jet_tau3 != 0 else 0
        jet_tau34 = jet_tau3 / jet_tau4 if jet_tau4 != 0 else 0

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
datos_HToBB = cargar_datos_json('./HToBB_120_flat.json', num_jets=10000, num_constituents=10)
datos_TTBar = cargar_datos_json('./TTBar_120_flat.json', num_jets=10000, num_constituents=10)
datos_WToqq = cargar_datos_json('./WToQQ_120_flat.json', num_jets=10000, num_constituents=10)

datos = []

def cargar_datos_json(json_path, num_jets=22500, num_constituents=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    eventos = []
    for i, evento in enumerate(data[:num_jets]):
        jet_pt, jet_eta, jet_phi, jet_mass = evento['jet_kinematics']
        constituents = evento['PFCands']
        theta = 2 * np.arctan(np.exp(-jet_eta))
        p = jet_pt / np.sin(theta)
        pz  = p * np.cos(theta)
        jet_energy = np.sqrt(pz**2 + jet_pt**2 + jet_mass**2)
        

        # Calculate pT for each constituent
        constituents = np.array(constituents)
        px = constituents[:, 0]
        py = constituents[:, 1]
        pt = np.sqrt(px**2 + py**2)

        # Indexes of the top num_constituents by pT
        indices_ordenados = np.argsort(pt)[::-1][:num_constituents]
        top_cands = constituents[indices_ordenados]

        # Convert each to the format used in the circuit
        top_constituents = []
        for cand in top_cands:
            px, py, pz, E = cand[0:4]
            d0 = cand[4]  # traversal impact parameter
            dz = cand[5]  # longitudinal impact parameter
            pt = np.sqrt(px**2 + py**2)
            p_total = np.sqrt(px**2 + py**2 + pz**2)
            eta = 0.5 * np.log((p_total + pz) / (p_total - pz + 1e-8))  # Avoiding dividing by 0
            phi = np.arctan2(py, px)
            mass = np.sqrt(np.maximum(0, E**2 - (px**2 + py**2 + pz**2)))# In case of negative mass, set it to zero
            top_constituents.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'mass': mass,
                'energy': E,
                'd0': d0,
                'dz': dz
            })

        eventos.append({
            'pt_jet': jet_pt,
            'eta_jet': jet_eta,
            'phi_jet': jet_phi,
            'mass_jet': jet_mass,
            'energy_jet': jet_energy,
            'constituents': top_constituents
        })

    return eventos



datos_00 = cargar_datos_json('./runG_batch0_flatpt.json')
datos_01 = cargar_datos_json('./runG_batch10_flatpt.json')
datos = datos_00 + datos_01


print('Datos cargados correctamente')

# Separar 10,000 para entrenamiento y 12,500 restantes
X_train, X_temp = train_test_split(
    datos, 
    train_size=10000, 
    random_state=42, 
    shuffle=True
)

# Separar 2,500 para validación y 10,000 para inferencia
X_val, rest = train_test_split(
    X_temp, 
    train_size=2500, 
    random_state=42, 
    shuffle=True
)

X_inf, rest = train_test_split(
    rest, 
    train_size=10000, 
    random_state=42, 
    shuffle=True
)

print("------------------------------")
print(f"Entrenamiento: {len(X_train)}")
print(f"Validación: {len(X_val)}")
print(f"Inferencia: {len(X_inf)}")
print("------------------------------")


Lambda = {
    1: torch.tensor([[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]], dtype=torch.cdouble),

    2: torch.tensor([[0, -1j, 0],
                 [1j, 0, 0],
                 [0, 0, 0]], dtype=torch.cdouble),

    3: torch.tensor([[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, 0]], dtype=torch.cdouble),

    4: torch.tensor([[0, 0, 1],
                 [0, 0, 0],
                 [1, 0, 0]], dtype=torch.cdouble),

    5: torch.tensor([[0, 0, -1j],
                 [0, 0, 0],
                 [1j, 0, 0]], dtype=torch.cdouble),

    6: torch.tensor([[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0]], dtype=torch.cdouble),

    7: torch.tensor([[0, 0, 0],
                 [0, 0, -1j],
                 [0, 1j, 0]], dtype=torch.cdouble),

    8: (1/torch.sqrt(torch.tensor(3.0))) * torch.tensor([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -2]], dtype=torch.cdouble),
    0: torch.tensor([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]], dtype=torch.cdouble)
}

Sigma = {
    1: (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.cdouble),

    2: (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor([
        [0, -1j, 0],
        [1j, 0, -1j],
        [0, 1j, 0]
    ], dtype=torch.cdouble),

    3: torch.tensor([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ], dtype=torch.cdouble)
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
    Gamma= 0
    a0 = 0
    a1 = 0
    a2 = 0

    Gamma = torch.sqrt(torch.tensor(2.0)) * (torch.tensor(3.0) + torch.cos(theta1)*torch.cos(theta2) + torch.sin(theta1)*torch.sin(theta2)*torch.cos(phi1 - phi2))**(torch.tensor(-0.5))

    a0 = (torch.sqrt(torch.tensor(2.0)) * torch.cos(theta1/2) * torch.cos(theta2/2)).item()
    a1 = (torch.exp(1j * phi1) * torch.sin(theta1/2) * torch.cos(theta2/2) + torch.cos(theta1/2) * torch.sin(theta2/2) * torch.exp(1j * phi2)).item()
    a2 = (torch.sqrt(torch.tensor(2.0)) * torch.exp(1j * (phi1 + phi2)) * torch.sin(theta1/2) * torch.sin(theta2/2)).item()

    state = Gamma * torch.tensor([a0, a1, a2], dtype=torch.cdouble)
    state = state / torch.linalg.norm(state)
    
    

    return state.detach().clone().numpy()



def unitary_from_state(psi):
    psi = psi / np.linalg.norm(psi)  # por seguridad
    
    a1 = torch.tensor([0.555,0, 0.555], dtype=torch.cdouble)
    a2 = torch.tensor([0.555,0.555, 0], dtype=torch.cdouble)
    mat = np.column_stack([psi, a1 , a2])

    Q, R = qr(mat)
    phase = np.vdot(psi, Q[:,0])
    Q[:,0] = Q[:,0] * (phase/abs(phase)).conj()
    
    return Q

# Parameters
from autoray import do


num_particles = 4
num_latent = 1
num_ref = num_particles - num_latent
num_trash = num_ref
wires = list(range(num_particles + num_ref + 1))  # +1 ancilla
ancilla = wires[-1]
dev = qml.device("default.qutrit", wires=wires)  

# Encoding functions
def f(w): return 1 + (2 * np.pi / (1 + torch.exp(-w)))
def phi_circuit(w, phi, phi_jet, pt, pt_jet): return f(w) * (pt / pt_jet) * (phi - phi_jet)
def theta_circuit(w, eta, eta_jet, pt, pt_jet): return f(w) * (pt / pt_jet) * (eta - eta_jet)
def mass_circuit(w, mass, mass_jet, pt, pt_jet):  return  f(w) * (pt / pt_jet) * (mass - mass_jet)
def energy_circuit(w, energy, energy_jet, pt, pt_jet): return f(w) * (pt / pt_jet) *  (energy - energy_jet)
def d0_circuit(w, d0, pt, pt_jet): return f(w) * (pt / pt_jet) * (d0)
def dz_circuit(w, dz, pt, pt_jet): return f(w) * (pt / pt_jet) * (dz)

# Encoding for qutrits
def encode_1p1q_qutrit(jets, w, unitaries):

    pt_jet = jets['pt_jet']
    eta_jet = jets['eta_jet']
    phi_jet = jets['phi_jet']
    mass_jet = jets['mass_jet']
    energy_jet = jets['energy_jet']
    constituents = jets['constituents']
        
    for i in range(num_particles):
        c = constituents[i]
        theta = theta_circuit(w, c['eta'], eta_jet, c['pt'], pt_jet)
        phi = phi_circuit(w, c['phi'], phi_jet, c['pt'], pt_jet)
        mass = mass_circuit(w, c['mass'], mass_jet, c['pt'], pt_jet)
        energy = energy_circuit(w, c['energy'], energy_jet, c['pt'], pt_jet)
        d0 = d0_circuit(w, c['d0'], c['pt'], pt_jet)
        dz = dz_circuit(w, c['dz'], c['pt'], pt_jet)

        initial_state = inicializing_qutrit_state( theta, phi, d0, dz)
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
def qae_circuit_qutrit(jets, w, theta_i, phi_i, w_i, num_layers):
    unitaries = []
    encode_1p1q_qutrit(jets, w, unitaries)
    variational_layer_qutrit(theta_i, phi_i, w_i, num_layers)
    tswap = TSWAP_matrix()

    for trash_wire, ref_wire in zip(trash_wires, ref_wires):
        qml.THadamard(wires=ancilla, subspace=None) #With none they apply the generalized version
        qml.ControlledQutritUnitary(tswap, control_wires=ancilla, wires=[trash_wire, ref_wire])
        qml.THadamard(wires=ancilla, subspace=None)
    
    return qml.probs(wires=ancilla)

def cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers):
    prob_0 = qae_circuit_qutrit(jet, w, theta_i, phi_i, w_i, num_layers)[0]
    fidelity = prob_0
    return -fidelity, fidelity.item()

fil_100_back = []
fil_100_HToBB = []
fil_100_WToQQ = []
fil_100_TTBar = []

for i in range(100):
    print("-----Inicio etapa ", i+1,"--------")
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
    event_fidelities = []  # List to store event fidelities

    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_fidelities = []
        avg_fidelity = 0.0
        avg_loss = 0.0
        
        for jet in X_train:
            if len(jet['constituents']) < num_particles:
                continue
        
            loss, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i,  num_layers)
            
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
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_back.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(0)

    

    inicio = time.time()
    for jet in datos_HToBB:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_HToBB.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    for jet in datos_TTBar:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_TTBar.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    for jet in datos_WToqq:
        if len(jet['constituents']) < num_particles:
            continue
        _, fidelity = cost_function_with_fidelity_qutrit(jet, w, theta_i, phi_i, w_i, num_layers)
        event_fidelities_WToQQ.append(fidelity * 100) 
        fidelidades.append(fidelity)
        etiquetas.append(1)

    

    fil_100_back.append(event_fidelities_back)
    fil_100_HToBB.append(event_fidelities_HToBB)
    fil_100_WToQQ.append(event_fidelities_WToQQ)
    fil_100_TTBar.append(event_fidelities_TTBar)

    print("------ Fin etapa ", i+1,"-----------")

fil_100_back_2=fil_100_back,
fil_100_HToBB_2=fil_100_HToBB,
fil_100_WToQQ_2=fil_100_WToQQ,
fil_100_TTBar_2=fil_100_TTBar

np.savez(
    'qutritsReal.npz',
    fil_100_back_S=fil_100_back_2,
    fil_100_HToBB_S=fil_100_HToBB_2,
    fil_100_WToQQ_S=fil_100_WToQQ_2,
    fil_100_TTBar_S=fil_100_TTBar_2
)
print("Guardado como qutritsReal.npz")

