import pytest
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.api.wavefunction_simulator_contracts import (
    simulator_contracts_for_tolerance,
    simulator_contracts_with_nontrivial_initial_state,
)
from orquestra.quantum.circuits import CNOT, Circuit, X
from qiskit_aer import Aer

from orquestra.integrations.qiskit.simulator import QiskitWavefunctionSimulator

STATEVECTOR_BACKENDS = ["aer_simulator_statevector"]


def _test_id(val):
    return val.backend.name()


@pytest.fixture(params=STATEVECTOR_BACKENDS)
def simulator(request):
    return QiskitWavefunctionSimulator(Aer.get_backend(request.param))


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_qiskit_wavefucntion_simulator_fulfills_circuit_runner_contracts(
    simulator, contract
):
    assert contract(simulator)


@pytest.mark.parametrize(
    "contract",
    simulator_contracts_for_tolerance()
    + simulator_contracts_with_nontrivial_initial_state(),
)
def test_qiskit_wf_simulator_fulfills_wf_simulator_contracts(simulator, contract):
    assert contract(simulator)


def test_running_batch_with_single_circuit_gives_correct_measurements(simulator):
    circuit = Circuit([X(0), CNOT(1, 2)])

    measurements_set = simulator.run_batch_and_measure([circuit], [100])

    assert len(measurements_set) == 1
    for measurements in measurements_set:
        assert len(measurements.bitstrings) == 100
        assert all(bitstring == (1, 0, 0) for bitstring in measurements.bitstrings)


def test_running_batch_with_many_circuits_gives_correct_measurements(simulator):
    n_circuits = 50
    n_samples = 100
    circuit = Circuit([X(0), CNOT(1, 2)])

    measurements_set = simulator.run_batch_and_measure(
        [circuit] * n_circuits, [n_samples] * n_circuits
    )

    assert len(measurements_set) == n_circuits

    for measurements in measurements_set:
        assert len(measurements.bitstrings) == n_samples
        assert all(bitstring == (1, 0, 0) for bitstring in measurements.bitstrings)
