import os

import pytest

from orquestra.integrations.qiskit.backend._ibmq_runner import create_ibmq_runner
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS


@pytest.fixture(scope="module")
def ibmq_runner():
    return create_ibmq_runner(
        api_token=os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1
    )


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_qiskit_runner_fulfills_circuit_runner_contracts(ibmq_runner, contract):
    assert contract(ibmq_runner)
