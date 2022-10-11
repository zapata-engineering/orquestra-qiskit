from functools import singledispatch
from typing import Union, Optional, Sequence

from qiskit import execute, ClassicalRegister, QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit_aer import AerProvider

from orquestra.integrations.qiskit.conversions import export_to_qiskit
from orquestra.quantum.api import BaseCircuitRunner
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.circuits._itertools import expand_sample_sizes, combine_measurements
from orquestra.quantum.measurements import Measurements

AnyQiskitBackend = Union[BackendV1, BackendV2]


def prepare_for_running_on_backend(circuit: Circuit) -> QuantumCircuit:
    qiskit_circuit = export_to_qiskit(circuit)
    qiskit_circuit.add_register(ClassicalRegister(size=qiskit_circuit.num_qubits))
    qiskit_circuit.measure(qiskit_circuit.qubits, qiskit_circuit.clbits)
    return qiskit_circuit


class QiskitRunner(BaseCircuitRunner):
    def __init__(self, qiskit_backend: AnyQiskitBackend):
        super().__init__()
        self.backend = qiskit_backend

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
        job = execute(
            new_circuits,
            backend=self.backend,
            shots=max(new_n_samples),
            optimization_level=0,
            backend_properties=self.backend.properties(),
        )

        return combine_measurements(job.result().get_counts(), multiplicities)
