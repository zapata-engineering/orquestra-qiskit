from functools import singledispatch
from typing import Union, Optional, Sequence, List

from qiskit import execute, ClassicalRegister, QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerProvider
from qiskit_aer.noise import NoiseModel

from orquestra.integrations.qiskit.conversions import export_to_qiskit
from orquestra.quantum.api import BaseCircuitRunner
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.circuits._itertools import expand_sample_sizes, combine_measurement_counts
from orquestra.quantum.circuits.layouts import CircuitConnectivity
from orquestra.quantum.measurements import Measurements

AnyQiskitBackend = Union[BackendV1, BackendV2]


def prepare_for_running_on_backend(circuit: Circuit) -> QuantumCircuit:
    qiskit_circuit = export_to_qiskit(circuit)
    qiskit_circuit.add_register(ClassicalRegister(size=qiskit_circuit.num_qubits))
    qiskit_circuit.measure(qiskit_circuit.qubits, qiskit_circuit.clbits)
    return qiskit_circuit


class QiskitRunner(BaseCircuitRunner):
    def __init__(
        self,
        qiskit_backend: AnyQiskitBackend,
        noise_model: Optional[NoiseModel] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,  # Throw away
        basis_gates: Optional[List[str]] = None,
        optimization_level: int = 0,
        seed: Optional[int] = None,
        execute_function=execute
    ):
        super().__init__()
        self.backend = qiskit_backend

        self.seed = seed
        self.backend = qiskit_backend
        self.noise_model = noise_model
        self.optimization_level = optimization_level

        self.basis_gates = (
            noise_model.basis_gates
            if basis_gates is None and noise_model is not None
            else basis_gates
        )

        self.device_connectivity = device_connectivity
        self._execute = execute_function

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        return self._run_batch_and_measure([circuit], [n_samples])[0]

    def _run_batch_and_measure(
        self, batch: Sequence[Circuit], samples_per_circuit: Sequence[int]
    ):
        circuits_to_execute = [
            prepare_for_running_on_backend(circuit) for circuit in batch
        ]

        new_circuits, new_n_samples, multiplicities = expand_sample_sizes(
            circuits_to_execute,
            samples_per_circuit,
            self.backend.configuration().max_shots,
        )

        coupling_map = (
            None if self.device_connectivity is None
            else
            CouplingMap(self.device_connectivity.connectivity)
        )

        job = self._execute(
            new_circuits,
            backend=self.backend,
            shots=max(new_n_samples),
            noise_model=self.noise_model,
            coupling_map=coupling_map,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

        all_counts = job.result().get_counts()
        # Qiskit backends return single dictionary with counts when there was
        # only one experiment. To simplify logic, we make sure to always have a
        # list.
        if not isinstance(all_counts, list):
            all_counts = [all_counts]

        return [
            Measurements.from_counts(counts)
            for counts in combine_measurement_counts(all_counts, multiplicities)
        ]
