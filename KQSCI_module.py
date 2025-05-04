# %%
import qiskit
import qiskit_aer
import qiskit_nature
import qiskit_nature_pyscf
import qiskit_algorithms
import qiskit_nature_pyscf
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer import  AerSimulator
# from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_algorithms.optimizers import SLSQP
from qiskit_aer.primitives import Estimator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from scipy.linalg import eigh
from qiskit_nature.second_q.circuit.library import HartreeFock
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
from numpy import pi
import pylatexenc
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------------

# %%
def read_states(file_path):
    state_probabilities = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_time = None
        for line in lines:
            line = line.strip()
            if line.startswith("Time:"):
                # Extract the time value
                current_time = float(line.split(":")[1].strip())
                state_probabilities[current_time] = {}
            elif current_time is not None and line:
                # Extract state and probability
                state, prob = line.split(":")
                state_probabilities[current_time][state.strip()] = float(prob.strip())
    return state_probabilities

# ---------------------------------------------------------------------------------------------------------------------------

def selection(state_probabilities, sample_space_size):
    result = {}
    print("Before normalising the states:")
    for time in sorted(state_probabilities.keys()):
        # Get the state probabilities for the current time
        probabilities = state_probabilities[time]

        # Sort states by their probabilities in descending order and take the top states
        top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:sample_space_size]

        # Store the top states and their probabilities in the result dictionary
        result[time] = top_states
    return result

# ---------------------------------------------------------------------------------------------------------------------------


# %%
def spin_adapted_selection(state_probabilities, num_qubit, sample_space_size):
    result = {}
    for time in sorted(state_probabilities.keys()):
        # Get the state probabilities for the current time
        probabilities = state_probabilities[time]

        # Sort states by their probabilities in descending order and take the top states
        top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:sample_space_size]

        # Dictionary to store possibly modified states
        modified_states = {}

        for dstates, prob in top_states:
            if dstates[:num_qubit] == dstates[num_qubit:]:
                # Symmetry adapted
                modified_states[dstates] = prob
            else:
                # Symmetry non-adapted: Adjust state
                newstate = dstates[num_qubit:] + dstates[:num_qubit]
                modified_states[dstates] = prob
                modified_states[newstate] = prob

        # Normalize the probabilities
        total_prob = sum(modified_states.values())
        modified_states = {state: (prob / total_prob) for state, prob in modified_states.items()}

        # Store the modified states for the current time
        result[time] = modified_states

    return result


# ---------------------------------------------------------------------------------------------------------------------------


# %%
from qiskit.quantum_info import Statevector
import numpy as np


def krylov_basis(result):
    combined_statevectors = {}

    for time, states_with_probs in result.items():
        # Initialize an empty state vector (assuming all states are of the same dimension)
        num_qubits = len(next(iter(states_with_probs.keys())))  # Length of a single state vector
        combined_vector = np.zeros(2**num_qubits, dtype=complex)
        combined_vector = Statevector(combined_vector)

        # Add the contributions of each selected state to the combined vector
        for state, probability in states_with_probs.items():
            combined_vector += probability**0.5 * Statevector.from_label(state)

        # Normalize the combined vector to ensure it represents a valid quantum state
        combined_vector /= np.linalg.norm(combined_vector)
        # Convert the combined vector into a Qiskit Statevector object
        combined_statevectors[time] = Statevector(combined_vector)
    
    return combined_statevectors
    # print(Statevector(combined_vector).expectation_value(s_squared))

# ---------------------------------------------------------------------------------------------------------------------------

# %%
def regularize_matrix(matrix, epsilon=1e-10):
    matrix += epsilon * np.eye(matrix.shape[0])
    return matrix


def kqsci_solver(qubit_hamiltonian, combined_statevectors, kqsci_size):
    # Construction of the matrix elements
    evolution_times = list(combined_statevectors.keys())
    E_sub = np.zeros((kqsci_size, kqsci_size), dtype=complex)
    S_sub = np.zeros((kqsci_size, kqsci_size), dtype=complex)
    for i in range(kqsci_size):
        for j in range(kqsci_size):
            state_evolve = combined_statevectors[evolution_times[j]].evolve(qubit_hamiltonian)
            state_inner_h = combined_statevectors[evolution_times[i]].inner(state_evolve)
            state_inner_s = combined_statevectors[evolution_times[i]].inner(combined_statevectors[evolution_times[j]])

            E_sub[i, j] = state_inner_h
            S_sub[i, j] = state_inner_s

    S_sub = regularize_matrix(S_sub)
    eigenvalues, eigenvectors = eigh(E_sub, S_sub)
    return eigenvalues, eigenvectors
# ---------------------------------------------------------------------------------------------------------------------------


# %%
def kqsci_eign(state_probabilities, num_qubits, sample_space_size, kqsci_size, qubit_hamiltonian):
    """
    Perform KQSCI eigenvalue and eigenvector computation.

    Parameters:
        state_probabilities (dict): Probabilities of evolved states.
        num_qubits (int): Number of qubits.
        sample_space_size (int): Size of the sample space.
        kqsci_size (int): Size of the KQSCI subspace.
        qubit_hamiltonian (SparsePauliOp): Qubit Hamiltonian.

    Returns:
        tuple: Eigenvalues and eigenvectors.
    """
    # Read the evolved states by application of unitaries by time evolution
    chis = spin_adapted_selection(state_probabilities, int(num_qubits / 2), sample_space_size)
    phis = krylov_basis(chis)
    
    # Call the solver function
    eigenvalues, eigenvectors = kqsci_solver(qubit_hamiltonian, phis, kqsci_size)
    
    return eigenvalues, eigenvectors
#%%
# ---------------------------------------------------------------------------------------------------------------------------
def kqsci_solver_full(qubit_hamiltonian, states, output_file):
    E_sub = np.zeros((len(states), len(states)), dtype=complex)
    S_sub = np.zeros((len(states), len(states)), dtype=complex)
    with open(output_file, "w") as file:
        for i in range(len(states)):
            for j in range(len(states)):
                state_evolve = Statevector.from_label(states[j]).evolve(qubit_hamiltonian)
                state_inner_h = (Statevector.from_label(states[i])).inner(state_evolve)
                state_inner_s = Statevector.from_label(states[i]).inner(Statevector.from_label(states[j]))

                E_sub[i, j] = state_inner_h
                S_sub[i, j] = state_inner_s
                file.write(f"{E_sub[i,j]} " )
                file.flush()
            file.write("\n")
    eigenvalues, eigenvectors = eigh(E_sub, S_sub)
    return eigenvalues, eigenvectors

#%%
# ---------------------------------------------------------------------------------------------------------------------------
def read_matrix_from_file(input_file, size=None):
    if size is None:
        raise ValueError("Matrix size must be provided to reconstruct the matrix.")

    E_sub = np.zeros((size, size), dtype=complex)
    with open(input_file, "r") as file:
        for i, line in enumerate(file):
            # Split the line into matrix elements
            elements = line.strip().split()
            for j, value in enumerate(elements):
                E_sub[i, j] = value

    return E_sub
#%%
# ---------------------------------------------------------------------------------------------------------------------------
from qiskit.quantum_info import Statevector
from scipy.linalg import eigh
from joblib import Parallel, delayed
import numpy as np

def compute_matrix_elements(i, j, states, qubit_hamiltonian):
    psi_i = Statevector.from_label(states[i])
    psi_j = Statevector.from_label(states[j])
    psi_j_evolved = psi_j.evolve(qubit_hamiltonian)

    H_elem = psi_i.inner(psi_j_evolved)
    S_elem = psi_i.inner(psi_j)
    return i, j, H_elem, S_elem

def kqsci_solver_full_parallel(qubit_hamiltonian, states, output_file, n_jobs=-1):
    n = len(states)
    E_sub = np.zeros((n, n), dtype=complex)
    S_sub = np.zeros((n, n), dtype=complex)

    # Parallel computation of all (i, j) elements
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_matrix_elements)(i, j, states, qubit_hamiltonian)
        for i in range(n) for j in range(n)
    )

    # Fill matrices from results
    for i, j, H_elem, S_elem in results:
        E_sub[i, j] = H_elem
        S_sub[i, j] = S_elem

    # Write to output file
    with open(output_file, "w") as file:
        for row in E_sub:
            file.write(" ".join(map(str, row)) + "\n")

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(E_sub, S_sub)
    return eigenvalues, eigenvectors
#%%
# ---------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import eigh

def orthonormalize_statevectors(statevectors):
    # Stack the statevectors into a matrix
    V = np.column_stack([sv.data for sv in statevectors])  # assuming sv.data is the raw vector
    Q, R = np.linalg.qr(V)  # QR decomposition
    return Q  # Orthonormal basis vectors as columns

def kqsci_solver_QR(qubit_hamiltonian, combined_statevectors, kqsci_size):
    # Prepare statevectors
    evolution_times = list(combined_statevectors.keys())
    raw_statevectors = [combined_statevectors[t] for t in evolution_times[:kqsci_size]]

    # Orthonormalize (QR or SVD can be used)
    ortho_basis_matrix = orthonormalize_statevectors(raw_statevectors)
    
    # Construct new Statevector objects from orthonormal basis
    from qiskit.quantum_info import Statevector
    ortho_statevectors = [Statevector(vec) for vec in ortho_basis_matrix.T]  # one per column

    # Construct subspace Hamiltonian and overlap matrices
    E_sub = np.zeros((kqsci_size, kqsci_size), dtype=complex)
    S_sub = np.zeros((kqsci_size, kqsci_size), dtype=complex)
    
    for i in range(kqsci_size):
        for j in range(kqsci_size):
            state_evolve = ortho_statevectors[j].evolve(qubit_hamiltonian)
            state_inner_h = ortho_statevectors[i].inner(state_evolve)
            state_inner_s = ortho_statevectors[i].inner(ortho_statevectors[j])

            E_sub[i, j] = state_inner_h
            S_sub[i, j] = state_inner_s

    # Overlap should now be close to identity, but regularize just in case
    # S_sub = regularize_matrix(S_sub)

    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(E_sub, S_sub)
    return eigenvalues, eigenvectors
#%%
# ---------------------------------------------------------------------------------------------------------------------------
def kqsci_eign_QR(state_probabilities, num_qubits, sample_space_size, kqsci_size, qubit_hamiltonian):
    """
    Perform KQSCI eigenvalue and eigenvector computation.

    Parameters:
        state_probabilities (dict): Probabilities of evolved states.
        num_qubits (int): Number of qubits.
        sample_space_size (int): Size of the sample space.
        kqsci_size (int): Size of the KQSCI subspace.
        qubit_hamiltonian (SparsePauliOp): Qubit Hamiltonian.

    Returns:
        tuple: Eigenvalues and eigenvectors.
    """
    # Read the evolved states by application of unitaries by time evolution
    chis = spin_adapted_selection(state_probabilities, int(num_qubits / 2), sample_space_size)
    phis = krylov_basis(chis)
    
    # Call the solver function
    eigenvalues, eigenvectors = kqsci_solver_QR(qubit_hamiltonian, phis, kqsci_size)
    
    return eigenvalues, eigenvectors
#%%
# ---------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# from scipy.linalg import eigh
# from qiskit.quantum_info import Statevector

def orthonormalize_statevectors_svd(statevectors, threshold=1e-8):
    # Stack the statevectors into a matrix (columns are state vectors)
    V = np.column_stack([sv.data for sv in statevectors])  # assumes Statevector.data is the raw numpy vector

    # Perform SVD
    U, S_vals, Vh = np.linalg.svd(V, full_matrices=False)

    # Filter by singular values (keep only numerically independent directions)
    rank = np.sum(S_vals > threshold)
    U_reduced = U[:, :rank]

    return U_reduced, rank  # Return orthonormal basis and its size

def kqsci_solver_SVD(qubit_hamiltonian, combined_statevectors, kqsci_size):
    # Get raw statevectors from dictionary
    evolution_times = list(combined_statevectors.keys())
    raw_statevectors = [combined_statevectors[t] for t in evolution_times[:kqsci_size]]

    # Orthonormalize using SVD
    ortho_basis_matrix, rank = orthonormalize_statevectors_svd(raw_statevectors)

    # Convert each orthonormal column vector into a Statevector
    ortho_statevectors = [Statevector(ortho_basis_matrix[:, i]) for i in range(rank)]

    # Construct Hamiltonian and overlap matrices (size rank x rank)
    E_sub = np.zeros((rank, rank), dtype=complex)
    S_sub = np.zeros((rank, rank), dtype=complex)

    for i in range(rank):
        for j in range(rank):
            state_evolve = ortho_statevectors[j].evolve(qubit_hamiltonian)
            state_inner_h = ortho_statevectors[i].inner(state_evolve)
            state_inner_s = ortho_statevectors[i].inner(ortho_statevectors[j])

            E_sub[i, j] = state_inner_h
            S_sub[i, j] = state_inner_s

    # Overlap matrix should now be very close to identity, but regularize just in case
    # S_sub = regularize_matrix(S_sub)

    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(E_sub, S_sub)
    return eigenvalues, eigenvectors
#%%
# ---------------------------------------------------------------------------------------------------------------------------
def kqsci_eign_SVD(state_probabilities, num_qubits, sample_space_size, kqsci_size, qubit_hamiltonian):
    """
    Perform KQSCI eigenvalue and eigenvector computation.

    Parameters:
        state_probabilities (dict): Probabilities of evolved states.
        num_qubits (int): Number of qubits.
        sample_space_size (int): Size of the sample space.
        kqsci_size (int): Size of the KQSCI subspace.
        qubit_hamiltonian (SparsePauliOp): Qubit Hamiltonian.

    Returns:
        tuple: Eigenvalues and eigenvectors.
    """
    # Read the evolved states by application of unitaries by time evolution
    chis = spin_adapted_selection(state_probabilities, int(num_qubits / 2), sample_space_size)
    phis = krylov_basis(chis)
    
    # Call the solver function
    eigenvalues, eigenvectors = kqsci_solver_SVD(qubit_hamiltonian, phis, kqsci_size)
    
    return eigenvalues, eigenvectors