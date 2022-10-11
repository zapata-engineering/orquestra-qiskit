import os

import pytest
from qiskit import Aer, QiskitError

from orquestra.integrations.qiskit.backend import QiskitRunner, QiskitWavefunctionSimulator
from orquestra.integrations.qiskit.noise import get_qiskit_noise_model
from orquestra.quantum.api import EstimationTask
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.circuits import Circuit, H, X
from orquestra.quantum.estimation import estimate_expectation_values_by_averaging
from orquestra.quantum.measurements import ExpectationValues
from orquestra.quantum.operators import PauliTerm


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
    "aer_simulator_extended_stabilizer",
    # "aer_simulator_unitary",             # Does not support measurements
    # "aer_simulator_superop",             # Does not support measurements
    "qasm_simulator",
    "statevector_simulator",
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
    return QiskitRunner(
        backend, noise_model=noise_model, device_connectivity=connectivity
    )


def test_initializing_simulator_with_noise_initializes_connectivity_and_basis(
    noisy_simulator
):
    assert noisy_simulator.device_connectivity is not None
    assert noisy_simulator.basis_gates is not None


@pytest.mark.parametrize("num_flips", [10, 50])
def test_expectation_value_with_noisy_simulator(noisy_simulator, num_flips):
    # Initialize in |1> state and flip even number of times.
    # Thus, we and up in |1> state but decoherence is allowed to take effect
    circuit = Circuit([X(0) for _ in range(num_flips+1)])
    qubit_operator = PauliTerm("Z0")
    n_samples = 8192

    estimation_tasks = [EstimationTask(qubit_operator, circuit, n_samples)]

    expectation_values = estimate_expectation_values_by_averaging(
        noisy_simulator, estimation_tasks
    )[0]

    assert isinstance(expectation_values, ExpectationValues)
    assert len(expectation_values.values) == 1
    assert -1 < expectation_values.values[0] < 0.0
