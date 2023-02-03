import os
from functools import partial
from unittest.mock import Mock

import pytest
from orquestra.quantum.api import EstimationTask
from orquestra.quantum.api.circuit_runner_contracts import (
    CIRCUIT_RUNNER_CONTRACTS,
    circuit_runner_gate_compatibility_contracts,
)
from orquestra.quantum.circuits import CNOT, Circuit, H, X
from orquestra.quantum.estimation import estimate_expectation_values_by_averaging
from orquestra.quantum.measurements import ExpectationValues
from orquestra.quantum.operators import PauliTerm
from qiskit import QiskitError, execute
from qiskit.transpiler import CouplingMap
from qiskit_aer import Aer

from orquestra.integrations.qiskit.noise import get_qiskit_noise_model
from orquestra.integrations.qiskit.runner import QiskitRunner


def _test_id(val):
    return val.backend.name()


@pytest.fixture
def aer_backend_with_real_shots_limit():
    """Sophisticated AerBackend mock that actually respects reported limits."""

    backend = Aer.get_backend("aer_simulator")
    max_shots = backend.configuration().max_shots
    old_run = backend.run

    def _run(*args, **kwargs):
        if kwargs.get("shots", 0) > max_shots:
            raise QiskitError("Maximum number of shots in experiment_exceeded")
        return old_run(*args, **kwargs)

    backend.run = _run
    return backend


COMPATIBLE_BACKENDS = (
    # This list contains names of all possible AerBackends.
    # The ones that are not working are commented out so that we
    # remember why they are not used in the tests
    "aer_simulator",
    "aer_simulator_statevector",
    "aer_simulator_density_matrix",
    # "aer_simulator_stabilizer",          # Has limited gateset
    "aer_simulator_matrix_product_state",
    # "aer_simulator_extended_stabilizer", # Technically compatible, incredibly slow
    # "aer_simulator_unitary",             # Does not support measurements
    # "aer_simulator_superop",             # Does not support measurements
    # "qasm_simulator",                    # Compatible, but deprecated
    # "statevector_simulator",             # Compatible, but deprecated
    # "unitary_simulator",                 # Does not support measurements
    # "pulse_simulator"                    # Does not support measurements
)


@pytest.mark.parametrize(
    "runner",
    [*[QiskitRunner(Aer.get_backend(name)) for name in COMPATIBLE_BACKENDS]],
    ids=_test_id,
)
@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_qiskit_runner_fulfills_circuit_runner_contracts(runner, contract):
    assert contract(runner)


@pytest.mark.parametrize(
    "contract",
    circuit_runner_gate_compatibility_contracts(
        gates_to_exclude=["RH", "XY"], exp_val_spread=1.5
    ),
)
@pytest.mark.parametrize(
    "runner",
    [
        *[
            QiskitRunner(
                Aer.get_backend(name),
                execute_function=partial(execute, seed_simulator=1234),
            )
            for name in COMPATIBLE_BACKENDS
        ]
    ],
    ids=_test_id,
)
def test_qiskit_runner_uses_correct_gate_definitions(runner, contract):
    assert contract(runner)


def test_qiskit_runner_can_run_job_with_sample_size_exceeding_backends_limit(
    aer_backend_with_real_shots_limit,
):
    runner = QiskitRunner(aer_backend_with_real_shots_limit)
    max_shots = runner.backend.configuration().max_shots
    circuit = Circuit([H(0)])

    measurements = runner.run_and_measure(circuit, n_samples=max_shots + 1)
    assert len(measurements.bitstrings) >= max_shots + 1


@pytest.fixture(params=["aer_simulator"])
def noisy_simulator(request):
    ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
    noise_model, connectivity = get_qiskit_noise_model(
        "ibm_nairobi", api_token=ibmq_api_token
    )
    backend = Aer.get_backend(request.param)
    return QiskitRunner(backend, noise_model=noise_model, coupling_map=connectivity)


@pytest.mark.parametrize("num_flips", [10, 50])
def test_expectation_value_with_noisy_simulator(noisy_simulator, num_flips):
    # Initialize in |1> state and flip even number of times.
    # Thus, we and up in |1> state but decoherence is allowed to take effect
    circuit = Circuit([X(0) for _ in range(num_flips + 1)])
    qubit_operator = PauliTerm("Z0")
    n_samples = 8192

    estimation_tasks = [EstimationTask(qubit_operator, circuit, n_samples)]

    expectation_values = estimate_expectation_values_by_averaging(
        noisy_simulator, estimation_tasks
    )[0]

    assert isinstance(expectation_values, ExpectationValues)
    assert len(expectation_values.values) == 1
    assert -1 < expectation_values.values[0] < 0.0


def test_qiskit_runner_passes_coupling_map_to_execute_function():
    circuit = Circuit([X(0), CNOT(1, 2)])

    coupling_map = CouplingMap([(0, 2), (0, 1)])
    execute_func = Mock(wraps=execute)

    runner = QiskitRunner(
        Aer.get_backend("aer_simulator_statevector"),
        coupling_map=coupling_map,
        execute_function=execute_func,
    )

    runner.run_and_measure(circuit, n_samples=10)
    assert execute_func.call_args.kwargs["coupling_map"] == coupling_map


@pytest.mark.parametrize(
    "runner",
    [
        *[
            QiskitRunner(Aer.get_backend(name), discard_extra_measurements=True)
            for name in COMPATIBLE_BACKENDS
        ]
    ],
    ids=_test_id,
)
def test_qiskit_runner_discards_extra_measurements_exact_num_measurements_is_true(
    runner: QiskitRunner,
):
    circuits = [Circuit([X(0), CNOT(0, 1)])] * 3
    n_samples = [5, 10, 15]
    result = runner.run_batch_and_measure(circuits, n_samples=n_samples)

    assert [len(r.bitstrings) for r in result] == n_samples
