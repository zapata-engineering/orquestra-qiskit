import pytest
from qiskit import Aer

from orquestra.integrations.qiskit.backend._qiskit_statevector_simulator import \
    QiskitWavefunctionSimulator
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.api.gate_model_simulator_contracts import simulator_contracts_for_tolerance


COMPATIBLE_BACKENDS = ["aer_simulator_statevector", "statevector_simulator"]


def _test_id(val):
    return val.backend.name()


@pytest.mark.parametrize(
    "simulator",
    [QiskitWavefunctionSimulator(Aer.get_backend(name)) for name in COMPATIBLE_BACKENDS],
    ids=_test_id,
)
@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_qiskit_wavefucntion_simulator_fulfills_circuit_runner_contracts(
    simulator, contract
):
    assert contract(simulator)


@pytest.mark.parametrize(
    "simulator",
    [QiskitWavefunctionSimulator(Aer.get_backend(name)) for name in COMPATIBLE_BACKENDS],
    ids=_test_id,
)
@pytest.mark.parametrize("contract", simulator_contracts_for_tolerance())
def test_qiskit_wf_simulator_fulfills_wf_simulator_contracts(
    simulator, contract
):
    assert contract(simulator)
