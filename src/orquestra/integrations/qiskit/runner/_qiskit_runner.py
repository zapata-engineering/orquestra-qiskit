from typing import List, Optional, Sequence, Union

from orquestra.quantum.api import BaseCircuitRunner
from orquestra.quantum.circuits import (
    Circuit,
    combine_bitstrings,
    expand_sample_sizes,
    split_into_batches,
)
from orquestra.quantum.measurements import Measurements
from qiskit import ClassicalRegister, QuantumCircuit, execute
from qiskit.providers import BackendV1, BackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel

from orquestra.integrations.qiskit.conversions import export_to_qiskit

AnyQiskitBackend = Union[BackendV1, BackendV2]


def prepare_for_running_on_backend(circuit: Circuit) -> QuantumCircuit:
    qiskit_circuit = export_to_qiskit(circuit)
    qiskit_circuit.add_register(ClassicalRegister(size=qiskit_circuit.num_qubits))
    qiskit_circuit.measure(qiskit_circuit.qubits, qiskit_circuit.clbits)
    return qiskit_circuit


def _listify(counts):
    return counts if isinstance(counts, list) else [counts]


class QiskitRunner(BaseCircuitRunner):
    def __init__(
        self,
        qiskit_backend: AnyQiskitBackend,
        noise_model: Optional[NoiseModel] = None,
        coupling_map: Optional[CouplingMap] = None,
        basis_gates: Optional[List[str]] = None,
        optimization_level: int = 0,
        seed: Optional[int] = None,
        discard_extra_measurements: bool = False,
        execute_function=execute,
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

        self.coupling_map = coupling_map
        self._execute = execute_function
        self.discard_extra_measurements = discard_extra_measurements

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

        batch_size = getattr(
            self.backend.configuration(), "max_experiments", len(circuits_to_execute)
        )

        batches = split_into_batches(new_circuits, new_n_samples, batch_size)

        jobs = [
            self._execute(
                list(circuits),
                backend=self.backend,
                shots=n_samples,
                noise_model=self.noise_model,
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                optimization_level=self.optimization_level,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
                memory=True,
            )
            for circuits, n_samples in batches
        ]

        # Qiskit runners return single dictionary with counts when there was
        # only one experiment. To simplify logic, we make sure to always have a
        # list of counts from a job.

        # One can use job.result().get_counts() to get all counts, but
        # job.result().get_memory() does require index of the experiment
        # This is why the below list comprehension looks so clumsy.
        all_bitstrings = [
            job.result().get_memory(i)
            for job in jobs
            for i in range(len(job.result().results))
        ]

        combined_bitstrings = combine_bitstrings(all_bitstrings, multiplicities)

        if self.discard_extra_measurements:
            combined_bitstrings = [
                bitstrings[:n_samples]
                for bitstrings, n_samples in zip(
                    combined_bitstrings, samples_per_circuit
                )
            ]

        return [
            Measurements([tuple(map(int, b[::-1])) for b in bitstrings])
            for bitstrings in combined_bitstrings
        ]
