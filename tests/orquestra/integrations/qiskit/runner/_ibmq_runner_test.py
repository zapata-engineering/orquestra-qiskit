import os

import pytest
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.circuits import CNOT, Circuit, H

from orquestra.integrations.qiskit.runner import create_ibmq_runner


@pytest.fixture(scope="module")
def ibmq_runner():
    return create_ibmq_runner(
        api_token=os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1,
    )


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_ibmq_runner_fulfills_circuit_runner_contracts(ibmq_runner, contract):
    assert contract(ibmq_runner)


def test_ibmq_runner_can_run_batches_larger_then_natively_supported_by_backend(
    ibmq_runner,
):
    max_native_batch_size = ibmq_runner.backend.configuration().max_experiments

    circuits = [
        Circuit([H(0), CNOT(0, 1)])
        for _ in range(max_native_batch_size + max_native_batch_size // 2)
    ]

    result = ibmq_runner.run_batch_and_measure(circuits, 1000)
    assert len(result) == len(circuits)


def test_ibmq_runner_discards_extra_measurements_if_exact_num_measurements_is_true():
    ibmq_runner = create_ibmq_runner(
        api_token=os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1,
        discard_extra_measurements=True,
    )
    circuits = [Circuit([H(0), CNOT(0, 1)])] * 3
    n_samples = [5, 10, 15]
    result = ibmq_runner.run_batch_and_measure(circuits, n_samples=n_samples)

    assert [len(r.bitstrings) for r in result] == n_samples
