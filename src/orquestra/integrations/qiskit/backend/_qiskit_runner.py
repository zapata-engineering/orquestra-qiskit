from typing import Union

from qiskit import execute, ClassicalRegister
from qiskit.providers import BackendV1, BackendV2

from orquestra.integrations.qiskit.conversions import export_to_qiskit
from orquestra.quantum.api import BaseCircuitRunner
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.measurements import Measurements

AnyQiskitBackend = Union[BackendV1, BackendV2]


class QiskitRunner(BaseCircuitRunner):

    def __init__(
        self,
        qiskit_backend: AnyQiskitBackend
    ):
        super().__init__()
        self.backend = qiskit_backend

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        qiskit_circuit = export_to_qiskit(circuit)
        qiskit_circuit.barrier(qiskit_circuit.qubits)
        qiskit_circuit.add_register(ClassicalRegister(size=qiskit_circuit.num_qubits))
        qiskit_circuit.measure(qiskit_circuit.qubits, qiskit_circuit.clbits)

        job = execute(
            qiskit_circuit,
            backend=self.backend,
            shots=n_samples,
            optimization_level=0,
            backend_properties=self.backend.properties()
        )
        return Measurements(job.result().get_counts())
