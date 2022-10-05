################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import math
import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Set, Tuple

from orquestra.quantum.api.backend import QuantumBackend
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.measurements import Measurements
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import (
    IBMQAccountError,
    IBMQBackendJobLimitError,
    IBMQProviderError,
)
from qiskit.providers.ibmq.job import IBMQJob
from qiskit.result import Counts
from qiskit.utils.mitigation import CompleteMeasFitter, complete_meas_cal
from qiskit.utils.mitigation._filters import MeasurementFilter

from orquestra.integrations.qiskit.conversions import export_to_qiskit


class QiskitBackend(QuantumBackend):
    def __init__(
        self,
        device_name: str,
        hub: Optional[str] = "ibm-q",
        group: Optional[str] = "open",
        project: Optional[str] = "main",
        api_token: Optional[str] = None,
        readout_correction: Optional[bool] = False,
        optimization_level: Optional[int] = 0,
        retry_delay_seconds: Optional[int] = 60,
        retry_timeout_seconds: Optional[int] = 86400,
        n_samples_for_readout_calibration: Optional[int] = None,
        noise_inversion_method: str = "least_squares",
        **kwargs,
    ):
        """Get a qiskit QPU that adheres to the
        orquestra.quantum.api.backend.QuantumBackend

        qiskit currently offers 2 types of qasm simulators:
        1. qasm_simulator - a local simulator that is depreciated.
        2. IBMQ_qasm_simulator - a remote simulator.
        All implementation of qasm_simulator have been removed since it's depreciation
        but IBMQ_qasm_simulator is still tested by this module.

        Args:
            device_name: the name of the device
            hub: IBMQ hub
            group: IBMQ group
            project: IBMQ project
            api_token: IBMQ Api Token
            readout_correction: flag of whether or not to use basic readout correction
            optimization_level: optimization level for the default qiskit transpiler (0,
                1, 2, or 3).
            retry_delay_seconds: Number of seconds to wait to resubmit a job when
                backend job limit is reached.
            retry_timeout_seconds: Number of seconds to wait
            noise_inversion_method (str): Method for inverting noise using readout
                correction. Options are "least_squares" and "pseudo_inverse".
                Defaults to "least_squares."
        """
        super().__init__()
        self.device_name = device_name

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if (
                    e.message
                    != "An IBM Quantum Experience account is already in use for the"
                    " session."
                ):
                    raise RuntimeError(e)

        try:
            provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        except IBMQProviderError as e:
            if api_token is None:
                raise RuntimeError("No providers were found. Missing IBMQ API token?")
            else:
                raise RuntimeError(e)

        self.device = provider.get_backend(name=self.device_name)
        self.max_shots = self.device.configuration().max_shots
        self.batch_size: int = self.device.configuration().max_experiments
        self.supports_batching = True
        self.readout_correction = readout_correction
        self.readout_correction_filters: Dict[str, MeasurementFilter] = {}
        self.optimization_level = optimization_level
        self.basis_gates = kwargs.get(
            "basis_gates", self.device.configuration().basis_gates
        )
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_timeout_seconds = retry_timeout_seconds
        self.n_samples_for_readout_calibration = n_samples_for_readout_calibration
        self.noise_inversion_method = noise_inversion_method
        self.list_virtual_to_physical_qubits_dict: List[Dict[int, int]] = []

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings.

        Args:
            circuit: the circuit to prepare the state
            n_samples: The number of samples to collect.
        """
        if n_samples <= 0:
            raise ValueError("n_samples should be greater than 0.")
        return self.run_circuitset_and_measure([circuit], [n_samples])[0]

    def run_circuitset_and_measure(
        self,
        circuits: Sequence[Circuit],
        n_samples: Sequence[int],
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.

        Args:
            circuitset: the circuits to run
            n_samples: The number of shots to perform on each circuit.

        Returns:
            A list of Measurements objects containing the observed bitstrings.
        """

        (
            experiments,
            n_samples_for_experiments,
            multiplicities,
        ) = self.transform_circuitset_to_ibmq_experiments(circuits, n_samples)
        (
            batches,
            n_samples_for_batches,
        ) = self.batch_experiments(experiments, n_samples_for_experiments)

        jobs = [
            self.execute_with_retries(batch, n_samples)
            for n_samples, batch in zip(n_samples_for_batches, batches)
        ]

        self.number_of_circuits_run += len(circuits)
        self.number_of_jobs_run += len(batches)

        return self.aggregate_measurements(jobs, batches, multiplicities)

    def transform_circuitset_to_ibmq_experiments(
        self,
        circuitset: Sequence[Circuit],
        n_samples: Sequence[int],
    ) -> Tuple[List[QuantumCircuit], List[int], List[int]]:
        """Convert circuits to qiskit and duplicate those whose measurement
        count exceeds the maximum allowed by the backend.

        Args:
            circuitset: The circuits to be executed.
            n_samples: A list of the number of samples to be collected for each
                circuit.

        Returns:
            Tuple containing:
            - The expanded list of circuits, converted to qiskit and each
              assigned a unique name.
            - List of number of samples for each element in expanded list of circuits
            - An array indicating how many duplicates there are for each of the
              original circuits.
        """
        ibmq_circuitset = []
        n_samples_for_ibmq_circuits = []
        multiplicities = []

        for n_samples_for_circuit, circuit in zip(n_samples, circuitset):
            ibmq_circuit = export_to_qiskit(circuit)
            full_qubit_indices = list(range(circuit.n_qubits))
            ibmq_circuit.barrier(full_qubit_indices)
            ibmq_circuit.add_register(ClassicalRegister(size=circuit.n_qubits))
            ibmq_circuit.measure(full_qubit_indices, full_qubit_indices)

            multiplicities.append(math.ceil(n_samples_for_circuit / self.max_shots))

            for i in range(multiplicities[-1]):
                ibmq_circuitset.append(ibmq_circuit.copy(f"{ibmq_circuit.name}_{i}"))

            for i in range(math.floor(n_samples_for_circuit / self.max_shots)):
                n_samples_for_ibmq_circuits.append(self.max_shots)

            if n_samples_for_circuit % self.max_shots != 0:
                n_samples_for_ibmq_circuits.append(
                    n_samples_for_circuit % self.max_shots
                )
        return ibmq_circuitset, n_samples_for_ibmq_circuits, multiplicities

    def batch_experiments(
        self,
        experiments: List[QuantumCircuit],
        n_samples_for_ibmq_circuits: List[int],
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

        batches: List = []
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
                            min(
                                len(batches) * self.batch_size,
                                len(experiments),
                            ),
                        )
                    ]
                )
            )

        return batches, n_samples_for_batches

    def execute_with_retries(
        self, batch: List[QuantumCircuit], n_samples: int
    ) -> IBMQJob:
        """Execute a job, resubmitting if the the backend job limit has been
        reached.

        The number of seconds between retries is specified by
        self.retry_delay_seconds. If self.retry_timeout_seconds is defined, then
        an exception will be raised if the submission does not succeed in the
        specified number of seconds.

        Args:
            batch: The batch of qiskit circuits to be executed.
            n_samples: The number of shots to perform on each circuit.

        Returns:
            The qiskit representation of the submitted job.
        """

        start_time = time.time()
        while True:
            try:
                job = execute(
                    batch,
                    self.device,
                    shots=n_samples,
                    basis_gates=self.basis_gates,
                    optimization_level=self.optimization_level,
                    backend_properties=self.device.properties(),
                )
                return job
            except IBMQBackendJobLimitError:
                if self.retry_timeout_seconds is not None:
                    elapsed_time_seconds = time.time() - start_time
                    if elapsed_time_seconds > self.retry_timeout_seconds:
                        raise RuntimeError(
                            f"Failed to submit job in {elapsed_time_seconds}s due to "
                            "backend job limit."
                        )
                print(f"Job limit reached. Retrying in {self.retry_delay_seconds}s.")
                time.sleep(self.retry_delay_seconds)  # type: ignore

    def aggregate_measurements(
        self,
        jobs: List[IBMQJob],
        batches: List[List[QuantumCircuit]],
        multiplicities: List[int],
    ) -> List[Measurements]:
        """Combine samples from a circuit set that has been expanded and batched
        to obtain a set of measurements for each of the original circuits. Also
        applies readout correction after combining.

        Args:
            jobs: The submitted IBMQ jobs.
            batches: The batches of experiments submitted.
            multiplicities: The number of copies of each of the original
                circuits.

        Returns:
            A list of list of measurements, where each list of measurements
            corresponds to one of the circuits of the original (unexpanded)
            circuit set.
        """
        circuit_set_from_jobs = []  # circuits that qiskit ran
        circuit_set_from_batches = []  # circuits that users sent
        self.list_virtual_to_physical_qubits_dict = []
        circuit_counts_set = []
        for job, batch in zip(jobs, batches):
            circuit_set_from_jobs.extend(job.circuits())
            circuit_set_from_batches.extend(batch)
            for experiment in batch:
                circuit_counts_set.append(job.result().get_counts(experiment))

        measurements_set = []
        circuit_index = 0
        for multiplicity in multiplicities:
            combined_counts = Counts({})
            for _ in range(multiplicity):
                for bitstring, counts in circuit_counts_set[circuit_index].items():
                    combined_counts[bitstring] = (
                        combined_counts.get(bitstring, 0) + counts
                    )
                circuit_index += 1

            current_circuit_from_jobs = circuit_set_from_jobs[circuit_index - 1]
            current_circuit_from_batches = circuit_set_from_batches[circuit_index - 1]
            virtual_to_physical_qubits_dict = _get_virtual_to_physical_qubits_dict(
                current_circuit_from_batches, current_circuit_from_jobs
            )
            if self.readout_correction:
                combined_counts = self._apply_readout_correction(
                    combined_counts, virtual_to_physical_qubits_dict
                )

            # qiskit counts object maps bitstrings in reversed order to ints, so we must
            # flip the bitstrings
            reversed_counts = {}
            for bitstring in combined_counts.keys():
                reversed_counts[bitstring[::-1]] = int(combined_counts[bitstring])

            measurements = Measurements.from_counts(reversed_counts)
            self.list_virtual_to_physical_qubits_dict.append(
                virtual_to_physical_qubits_dict
            )
            measurements_set.append(measurements)

        return measurements_set

    def _apply_readout_correction(
        self,
        counts: Counts,
        virtual_to_physical_qubits_dict: Optional[Dict[int, int]] = None,
    ):
        """Returns the counts from an experiment with readout correction applied to a
        set of qubits labeled physical_qubits. Output counts will only show outputs for
        corrected qubits. If no filter exists for the current physical_qubits the
        function will make one. Otherwise, function will re-use filter it created
        for these physical_qubits previously. Has 8 digits of precision.

        Args:
            counts (Counts): Dictionary containing the number of times a bitstring
                was received in an experiment.
            virtual_to_physical_qubits_dict (Optional[Dict[int, int]], optional):
                a dictionary that when given a virtual qubit,
                returns the corresponding physical qubit.
                Defaults to readout correction on all qubits.

        Raises:
            TypeError: If n_samples_for_readout_correction was not defined when the
                QiskitBackend Object was declared.

        Returns:
            mitigated_counts (Counts): counts for each output bitstring only showing
                the qubits which were mitigated.
        """

        for key in counts.keys():
            num_qubits = len(key)
            break

        if virtual_to_physical_qubits_dict is None:
            virtual_qubits = list(range(num_qubits))
            physical_qubits = list(range(num_qubits))
        else:
            virtual_qubits = list(virtual_to_physical_qubits_dict.keys())
            virtual_qubits.sort()
            physical_qubits = [
                virtual_to_physical_qubits_dict[virtual_qubit]
                for virtual_qubit in virtual_qubits
            ]
            for key in deepcopy(list(counts.keys())):
                new_key = "".join(key[num_qubits - i - 1] for i in virtual_qubits)
                counts[new_key] = counts.pop(key) + counts.get(new_key, 0)

        if not self.readout_correction_filters.get(str(physical_qubits)):

            if self.n_samples_for_readout_calibration is None:
                raise TypeError(
                    "n_samples_for_readout_calibration must"
                    "be set to use readout calibration"
                )

            qr = QuantumRegister(num_qubits)
            meas_cals, state_labels = complete_meas_cal(
                qubit_list=physical_qubits, qr=qr
            )

            # Execute the calibration circuits
            job = self.execute_with_retries(
                meas_cals, self.n_samples_for_readout_calibration
            )
            cal_results = job.result()

            # Make a calibration matrix
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)

            # Create a measurement filter from the calibration matrix
            self.readout_correction_filters[str(physical_qubits)] = meas_fitter.filter

        this_filter = self.readout_correction_filters[str(physical_qubits)]
        mitigated_counts = this_filter.apply(counts, method=self.noise_inversion_method)
        # round to make up for precision loss from pseudoinverses used to invert noise
        rounded_mitigated_counts = {
            k: round(v, 8) for k, v in mitigated_counts.items() if round(v, 8) != 0
        }
        return rounded_mitigated_counts


def _get_active_qubits(circuit: QuantumCircuit) -> Set[int]:
    """Returns a list of qubits that qiskit gates are operating on (i.e., active qubits).

    Args:
        circuit (QuantumCircuit): a circuit that we want to find the active qubits for.

    Returns:
        active_qubits (Set[int]): a list of active qubits.
    """
    qreg_size = sum(len(qreg) for qreg in circuit.qregs)
    active_qubits: Set = set()

    for inst in circuit.data:
        if inst[0].name not in [
            "measure",
            "barrier",
        ]:  # if not measure/barrier, then it is a Quantum Gate
            active_qubits = active_qubits.union({q.index for q in inst[1]})
            if len(active_qubits) == qreg_size:
                break

    return active_qubits


def _get_clbit_qubit_map(circuit: QuantumCircuit) -> List[int]:
    """Returns a list of qubits where
        their indices are the corresponding classical bits.

    Args:
        circuit (QuantumCircuit): a circuit for which
            we want to find the clbit-to-qubit map.

    Raises:
        ValueError: If the QuantumCircuit has more than 1 ClassicalRegister.
        AssertionError: If the number of measure operations
            is less than the number of classical bits.

    Returns:
        clbit_qubit_map (List[int]): a list of qubits
            whose indices correspond to their classical bits.
    """
    cregs_list = circuit.cregs

    num_cregs = len(cregs_list)
    if num_cregs > 1:
        raise ValueError(
            f"QuantumCircuit has {num_cregs} ClassicalRegister."
            "Currently, QuantumCircuit with only 1 ClassicalRegister is supported."
        )

    creg_size = cregs_list[0].size  # potential number of measured qubits

    clbit_qubit_map = [-1] * creg_size

    measure_op_found = 0

    """Each item in QuantumCircuit.data is a tuple:
        (<instuction name@idx=0>, <quantum register@idx=1>, <classcical register@idx=2>)
    """
    inst_idx = 0
    qreg_idx = 1
    creg_idx = 2

    """The following for loop gets the `instructions` of a QuantumCircuit from
    reverese (as `measure` instructions are usually at the end of a QuantumCircuit).
    The loop breaks when all the `measure` instructions are found.
    """
    for inst in circuit.data[::-1]:
        if inst[inst_idx].name == "measure":
            measure_op_found += 1

            qubit = inst[qreg_idx][0].index  # can be physical qubit or virtual qubit
            clbit = inst[creg_idx][0].index

            clbit_qubit_map[clbit] = qubit

            if measure_op_found == creg_size:
                break

    assert measure_op_found == creg_size, (
        f"measured op found {measure_op_found} is less than ClassicalRegister size"
        f" {creg_size}"
    )

    return clbit_qubit_map


def _get_virtual_to_physical_qubits_dict(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> Dict[int, int]:
    """Returns a dictionary that
        when given a qubit from the original circuit (i.e., a virtual qubit),
        returns a qubit from the transpiled circuit (i.e., a physical qubit).

    Args:
        original_circuit (QuantumCircuit): a circuit that
            we submitted to qiskit and from which we get our virtual qubits.
        transpiled_circuit (QuantumCircuit): a circuit that
            qiskit ran and from which we get our physical qubits.

    Raises:
        AssertionError: If the number of virtual qubits
            is not equal to the number of physical qubits.

    Returns:
        virtual_to_physical_qubits_dict (Dict[int, int]): a dictionary that
            when given a virtual qubit returns the corresponding physical qubit.
    """
    active_virtual_qubits = _get_active_qubits(
        original_circuit
    )  # we only use the active qubits for readout correction
    clbits_to_virtual_qubits = _get_clbit_qubit_map(original_circuit)
    clbits_to_physical_qubits = _get_clbit_qubit_map(transpiled_circuit)

    assert len(clbits_to_virtual_qubits) == len(clbits_to_physical_qubits)

    virtual_to_physical_qubits_dict_all = {
        c2v: c2p
        for c2v, c2p in zip(clbits_to_virtual_qubits, clbits_to_physical_qubits)
    }
    virtual_to_physical_qubits_dict = {
        virtual: physical
        for virtual, physical in virtual_to_physical_qubits_dict_all.items()
        if virtual in active_virtual_qubits
    }

    return virtual_to_physical_qubits_dict
