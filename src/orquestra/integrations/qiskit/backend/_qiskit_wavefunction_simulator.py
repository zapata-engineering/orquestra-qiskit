from typing import Optional, List

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel

from orquestra.integrations.qiskit.backend._qiskit_runner import AnyQiskitBackend
from orquestra.integrations.qiskit.conversions import export_to_qiskit
from orquestra.quantum.api.backend import StateVector
from orquestra.quantum.api.simulator import BaseWavefunctionSimulator
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.circuits.layouts import CircuitConnectivity
from orquestra.quantum.wavefunction import flip_amplitudes


class QiskitWavefunctionSimulator(BaseWavefunctionSimulator):

    def __init__(
        self,
        qiskit_backend: AnyQiskitBackend,
        noise_model: Optional[NoiseModel] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
        basis_gates: Optional[List[str]] = None,

        seed: Optional[int] = None
    ):
        super().__init__()
        self.seed = seed
        self.backend = qiskit_backend
        self.noise_model = noise_model

        self.basis_gates = (
            noise_model.basis_gates
            if basis_gates is None and noise_model is not None
            else basis_gates
        )

        self.device_connectivity = device_connectivity

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ):
        qiskit_circuit = export_to_qiskit(circuit)

        if not np.array_equal(initial_state, [1] + [0] * (2 ** circuit.n_qubits - 1)):
            state_prep_circuit = QuantumCircuit(circuit.n_qubits)
            state_prep_circuit.initialize(flip_amplitudes(initial_state))
            qiskit_circuit = state_prep_circuit.compose(qiskit_circuit)

        qiskit_circuit.save_state()

        coupling_map = (
            None if self.device_connectivity is None
            else
            CouplingMap(self.device_connectivity.connectivity)
        )

        job = execute(
            qiskit_circuit,
            self.backend,
            noise_model=self.noise_model,
            coupling_map=coupling_map,
            basis_gates=self.basis_gates,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

        return flip_amplitudes(job.result().get_statevector(
            qiskit_circuit, decimals=20
        ).data)