import pytest
from qiskit import Aer

from orquestra.integrations.qiskit.backend import QiskitWavefunctionSimulator
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.api.wavefunction_simulator_contracts import simulator_contracts_for_tolerance

STATEVECTOR_BACKENDS = ["aer_simulator_statevector", "statevector_simulator"]


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


@pytest.mark.parametrize("contract", simulator_contracts_for_tolerance())
def test_qiskit_wf_simulator_fulfills_wf_simulator_contracts(
    simulator, contract
):
    assert contract(simulator)
