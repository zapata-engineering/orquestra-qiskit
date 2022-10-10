import pytest
from qiskit import Aer

from orquestra.integrations.qiskit.backend import QiskitRunner
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.circuits import Circuit, H


def _test_id(val):
    return val.backend.name()


COMPATIBLE_BACKENDS = (
    # This list contains names of all possible AerBackends.
    # The ones that are not working are commented out so that we
    # remember why they are not used in the tests
    "aer_simulator",
    "aer_simulator_statevector",
    "aer_simulator_density_matrix",
    # "aer_simulator_stabilizer",          # Has limited gateset, cannot run some gates used in contracts
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
    [
        *[
            QiskitRunner(Aer.get_backend(name))
            for name in COMPATIBLE_BACKENDS
        ]
    ],
    ids=_test_id
)
@pytest.mark.parametrize(
    "contract",
    CIRCUIT_RUNNER_CONTRACTS
)
def test_qiskit_runner_fulfills_circuit_runner_contracts(runner, contract):
    assert contract(runner)

def test_qiskit_runner_can_run_job_with_sample_size_exceeding_backends_limit():
    backend = Aer.get_backend("aer_simulator_statevector")
    runner = QiskitRunner(backend)
    max_shots = backend.configuration().max_shots
    circuit = Circuit([H(0)])

    measurements = runner.run_and_measure(circuit, n_samples=max_shots+1)
    assert len(measurements.bitstrings) == max_shots + 1
