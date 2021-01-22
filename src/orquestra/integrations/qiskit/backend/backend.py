from qiskit import execute, QuantumRegister, QuantumCircuit
from qiskit.providers.ibmq import IBMQ
from qiskit.ignis.mitigation.measurement import (
    complete_meas_cal,
    CompleteMeasFitter,
)
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.result import Counts
from qiskit.providers.ibmq.job import IBMQJob
from openfermion.ops import IsingOperator
from zquantum.core.openfermion import change_operator_type
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.circuit import Circuit
from zquantum.core.measurement import (
    expectation_values_to_real,
    Measurements,
)
from typing import List, Optional, Tuple
import math


class QiskitBackend(QuantumBackend):
    def __init__(
        self,
        device_name,
        n_samples=None,
        hub="ibm-q",
        group="open",
        project="main",
        api_token=None,
        readout_correction=False,
        optimization_level=0,
        **kwargs,
    ):
        """Get a qiskit QPU that adheres to the
        zquantum.core.interfaces.backend.QuantumBackend

        Args:
            device_name (string): the name of the device
            n_samples (int): the number of samples to use when running the device
            hub (string): IBMQ hub
            group (string): IBMQ group
            project (string): IBMQ project
            api_token (string): IBMQ Api Token
            readout_correction (bool): indication of whether or not to use basic readout correction
            optimization_level (int): optimization level for the default qiskit transpiler (0, 1, 2, or 3)

        Returns:
            qeqiskit.backend.QiskitBackend
        """
        super().__init__(n_samples=n_samples)
        self.device_name = device_name

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if (
                    e.message
                    != "An IBM Quantum Experience account is already in use for the session."
                ):
                    raise RuntimeError(e)

        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        self.device = provider.get_backend(name=self.device_name)
        self.max_shots = self.device.configuration().max_shots
        self.batch_size = self.device.configuration().max_experiments
        self.supports_batching = True
        self.readout_correction = readout_correction
        self.readout_correction_filter = None
        self.optimization_level = optimization_level

    def run_circuit_and_measure(self, circuit, **kwargs):
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state

        Returns:
            a list of bitstrings (a list of tuples)
        """
        super().run_circuit_and_measure(circuit)
        num_qubits = len(circuit.qubits)

        ibmq_circuit = circuit.to_qiskit()
        ibmq_circuit.barrier(range(num_qubits))
        ibmq_circuit.measure(range(num_qubits), range(num_qubits))

        # Run job on device and get counts
        raw_counts = (
            execute(
                ibmq_circuit,
                self.device,
                shots=self.n_samples,
                optimization_level=self.optimization_level,
            )
            .result()
            .get_counts()
        )

        if self.readout_correction:
            raw_counts = self.apply_readout_correction(raw_counts, kwargs)

        # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
        reversed_counts = {}
        for bitstring in raw_counts.keys():
            reversed_counts[bitstring[::-1]] = int(raw_counts[bitstring])
        measurements = Measurements.from_counts(reversed_counts)

        return measurements

    def expand_circuitset(
        self, circuitset: List[Circuit], n_samples: Optional[List[int]] = None
    ) -> Tuple[List[QuantumCircuit], List[int], List[int]]:
        """Duplicate circuits whose requested number of measurements exceeds the
        maximum allowed by the backend.

        Args:
            circuitset: The circuits to be executed.
            n_samples: A list of the number of samples to be collected for each
                circuit. If None, self.n_samples is used for each circuit.
        
        Returns:
            Tuple containing:
            - The expanded list of circuits, converted to qiskit and each
              assigned a unique name.
            - An array indicating how many duplicates there are for each of the
              original circuits. 
        """
        ibmq_circuitset = []
        n_samples_for_ibmq_circuits = []
        multiplicities = []

        if not n_samples:
            n_samples = (self.n_samples,) * len(circuitset)

        for n_samples_for_circuit, circuit in zip(n_samples, circuitset):
            num_qubits = len(circuit.qubits)

            ibmq_circuit = circuit.to_qiskit()
            ibmq_circuit.barrier(range(num_qubits))
            ibmq_circuit.measure(range(num_qubits), range(num_qubits))

            multiplicities.append(math.ceil(n_samples_for_circuit / self.max_shots))

            for i in range(multiplicities[-1]):
                ibmq_circuitset.append(ibmq_circuit.copy(f"{ibmq_circuit.name}_{i}"))

            if math.floor(n_samples_for_circuit / self.max_shots) > 0:
                n_samples_for_ibmq_circuits.append(
                    self.max_shots * math.floor(n_samples_for_circuit / self.max_shots)
                )
            if n_samples_for_circuit % self.max_shots != 0:
                n_samples_for_ibmq_circuits.append(
                    n_samples_for_circuit % self.max_shots
                )
        return ibmq_circuitset, n_samples_for_ibmq_circuits, multiplicities

    def batch_experiments(
        self, experiments: List[QuantumCircuit], n_samples_for_ibmq_circuits: List[int],
    ) -> Tuple[List[List[QuantumCircuit]], List[int]]:
        """Batch a set of experiments (circuits to be executed) into groups
        whose size is no greater than the maximum allowed by the backend.

        Args:
            experiments: The circuits to be executed.
            n_samples_for_ibmq_circuits: The number of samples desired for each
                experiment.

        Returns:
            A tuple containing:
            - A list of batches, where each batch is a list of experiments.
            - An array containing the number of measurements that must be
              performed for each batch so that each experiment receives at least
              as many samples as specified by n_samples_for_ibmq_circuits.
        """

        batches = []
        n_samples_for_batches = []
        while len(batches) * self.batch_size < len(experiments):
            batches.append(
                [
                    experiments[i]
                    for i in range(
                        len(batches) * self.batch_size,
                        min(
                            len(batches) * self.batch_size + self.batch_size,
                            len(experiments),
                        ),
                    )
                ]
            )

            n_samples_for_batches.append(
                max(
                    [
                        n_samples_for_ibmq_circuits[i]
                        for i in range(
                            len(batches) * self.batch_size - self.batch_size,
                            min(len(batches) * self.batch_size, len(experiments),),
                        )
                    ]
                )
            )

        return batches, n_samples_for_batches

    def aggregregate_measurements(
        self,
        jobs: List[IBMQJob],
        batches: List[List[QuantumCircuit]],
        multiplicities: List[int],
        **kwargs,
    ) -> List[Measurements]:
        """Combine samples from a circuit set that has been expanded and batched
        to obtain a set of measurements for each of the original circuits. Also
        applies readout correction after combining.

        Args:
            jobs: The submitted IBMQ jobs.
            batches: The batches of experiments submitted.
            multiplicities: The number of copies of each of the original
                circuits.
            kwargs: Passed to self.apply_readout_correction.
        
        Returns:
            A list of list of measurements, where each list of measurements
            corresponds to one of the circuits of the original (unexpanded)
            circuit set. 
        """
        ibmq_circuit_counts_set = []
        for job, batch in zip(jobs, batches):
            for experiment in batch:
                ibmq_circuit_counts_set.append(job.result().get_counts(experiment))

        measurements_set = []
        ibmq_circuit_index = 0
        for multiplicity in multiplicities:
            combined_counts = Counts({})
            for i in range(multiplicity):
                for bitstring, counts in ibmq_circuit_counts_set[
                    ibmq_circuit_index
                ].items():
                    combined_counts[bitstring] = (
                        combined_counts.get(bitstring, 0) + counts
                    )
                ibmq_circuit_index += 1

            if self.readout_correction:
                combined_counts = self.apply_readout_correction(combined_counts, kwargs)

            # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
            reversed_counts = {}
            for bitstring in combined_counts.keys():
                reversed_counts[bitstring[::-1]] = int(combined_counts[bitstring])

            measurements = Measurements.from_counts(reversed_counts)
            measurements_set.append(measurements)

        return measurements_set

    def run_circuitset_and_measure(
        self, circuitset: List[Circuit], n_samples: Optional[List[int]] = None, **kwargs
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples

        Args:
            circuitset: the circuits to run
            n_samples: The number of shots to perform on each circuit. If
                None, then self.n_samples shots are performed for each circuit. 

        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """

        experiments, n_samples_for_experiments, multiplicities = self.expand_circuitset(
            circuitset, n_samples
        )
        batches, n_samples_for_batches = self.batch_experiments(
            experiments, n_samples_for_experiments
        )

        jobs = [
            execute(
                batch,
                self.device,
                shots=n_samples,
                optimization_level=self.optimization_level,
            )
            for n_samples, batch in zip(n_samples_for_batches, batches)
        ]

        self.number_of_circuits_run += len(circuitset)
        self.number_of_jobs_run += len(experiments)

        return self.aggregregate_measurements(jobs, batches, multiplicities)

    def apply_readout_correction(self, counts, qubit_list=None, **kwargs):
        if self.readout_correction_filter is None:

            for key in counts.keys():
                num_qubits = len(key)
                break

            if qubit_list is None or qubit_list == {}:
                qubit_list = [i for i in range(num_qubits)]

            qr = QuantumRegister(num_qubits)
            meas_cals, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr)

            # Execute the calibration circuits
            job = execute(meas_cals, self.device, shots=self.n_samples)
            cal_results = job.result()

            # Make a calibration matrix
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)
            # Create a measurement filter from the calibration matrix
            self.readout_correction_filter = meas_fitter.filter

        mitigated_counts = self.readout_correction_filter.apply(counts)
        return mitigated_counts
