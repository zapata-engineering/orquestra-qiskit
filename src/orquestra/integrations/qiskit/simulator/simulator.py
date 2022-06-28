################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import List, Optional

import numpy as np
from orquestra.quantum.api.backend import QuantumSimulator, StateVector
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.circuits.layouts import CircuitConnectivity
from orquestra.quantum.measurements import Measurements
from orquestra.quantum.wavefunction import flip_amplitudes, sample_from_wavefunction
from qiskit import Aer, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.transpiler import CouplingMap

from orquestra.integrations.qiskit.conversions import export_to_qiskit


class QiskitSimulator(QuantumSimulator):
    supports_batching = False
    batch_size = sys.maxsize

    def __init__(
        self,
        device_name: str,
        noise_model: Optional[NoiseModel] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
        basis_gates: Optional[List] = None,
        api_token: Optional[str] = None,
        optimization_level: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Get a qiskit device (simulator or QPU) that adheres to the
        orquestra.quantum.api.backend.QuantumSimulator

        Args:
            device_name: the name of the device
            noise_model: an optional
                noise model to pass in for noisy simulations
            device_connectivity: an optional input of an object representing
                the connectivity of the device that will be used in simulations
            basis_gates: an optional input of the list of basis gates
                used in simulations
            api_token: IBMQ Api Token
            optimization_level: optimization level for the default qiskit transpiler.
                It can take values 0, 1, 2 or 3.
            seed: seed for RNG
        """
        super().__init__()
        self.device_name = device_name
        self.noise_model = noise_model
        self.device_connectivity = device_connectivity
        self.seed = seed

        if basis_gates is None and self.noise_model is not None:
            self.basis_gates = self.noise_model.basis_gates
        else:
            self.basis_gates = basis_gates

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if (
                    e.message
                    != "An IBM Quantum Experience account is already in use for the session."  # noqa: E501
                ):
                    raise RuntimeError(e)

        self.optimization_level = optimization_level
        self.get_device(**kwargs)

    def get_device(self, noisy=False, **kwargs):
        """Get the ibm device used for executing circuits

        Args:
            noisy (bool): a boolean indicating if the user wants to use noisy
                simulations
        Returns:
            The ibm device that can use the ibm execute api
        """
        # If not doing noisy simulation...
        if len(Aer.backends(self.device_name)) > 0:
            self.device = Aer.get_backend(self.device_name)
        else:
            raise RuntimeError(
                "Could not find simulator with name: {}".format(self.device_name)
            )

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples

        Args:
            circuit: the circuit to prepare the state
            n_samples: The number of samples to collect.
        """

        super().run_circuit_and_measure(circuit, n_samples)
        num_qubits = circuit.n_qubits

        ibmq_circuit = export_to_qiskit(circuit)
        ibmq_circuit.barrier(range(num_qubits))
        ibmq_circuit.add_register(ClassicalRegister(size=circuit.n_qubits))
        ibmq_circuit.measure(range(num_qubits), range(num_qubits))

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        if self.device_name == "aer_simulator_statevector":
            wavefunction = self.get_wavefunction(circuit)
            return Measurements(sample_from_wavefunction(wavefunction, n_samples))
        else:
            # Run job on device and get counts
            raw_counts = (
                execute(
                    ibmq_circuit,
                    self.device,
                    shots=n_samples,
                    noise_model=self.noise_model,
                    coupling_map=coupling_map,
                    basis_gates=self.basis_gates,
                    optimization_level=self.optimization_level,
                    seed_simulator=self.seed,
                    seed_transpiler=self.seed,
                )
                .result()
                .get_counts()
            )

        # qiskit counts object maps bitstrings in reversed order to ints,
        # so we must flip the bitstrings
        reversed_counts = {}
        for bitstring in raw_counts.keys():
            reversed_counts[bitstring[::-1]] = raw_counts[bitstring]

        return Measurements.from_counts(reversed_counts)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        """Run a circuit and get the wavefunction of the resulting statevector.

        Args:
            circuit (orquestra.quantum.circuit.Circuit): the circuit to
                prepare the state.
        Returns:
            orquestra.quantum.wavefunction.Wavefunction
        """
        ibmq_circuit = export_to_qiskit(circuit)

        if not np.array_equal(initial_state, [1] + [0] * (2**circuit.n_qubits - 1)):
            state_prep_circuit = QuantumCircuit(circuit.n_qubits)
            state_prep_circuit.initialize(flip_amplitudes(initial_state))
            ibmq_circuit = state_prep_circuit.compose(ibmq_circuit)

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        if self.device_name == "aer_simulator_statevector":
            ibmq_circuit.save_state()

        # Execute job to get wavefunction
        job = execute(
            ibmq_circuit,
            self.device,
            noise_model=self.noise_model,
            coupling_map=coupling_map,
            basis_gates=self.basis_gates,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        wavefunction = job.result().get_statevector(ibmq_circuit, decimals=20)
        return flip_amplitudes(wavefunction.data)
