import os
from unittest.mock import Mock, create_autospec

import pytest
from orquestra.quantum.api.circuit_runner_contracts import CIRCUIT_RUNNER_CONTRACTS
from orquestra.quantum.circuits import CNOT, Circuit, H
from qiskit.result import Result
from qiskit_ibm_provider import IBMBackend, IBMBackendApiError, IBMJob, IBMProvider
from qiskit_ibm_provider.ibm_backend import QasmBackendConfiguration

from orquestra.integrations.qiskit.runner import create_ibmq_runner


@pytest.fixture(scope="module")
def ibmq_runner():
    return create_ibmq_runner(
        api_token=os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1,
    )


@pytest.fixture
def mock_execute(monkeypatch: pytest.MonkeyPatch):
    execute = Mock()
    monkeypatch.setattr(
        "orquestra.integrations.qiskit.runner._ibmq_runner.execute", execute
    )
    return execute


@pytest.fixture
def mock_ibm_backend(monkeypatch: pytest.MonkeyPatch):
    """This mocks enough of an IBM backend to run the tests"""
    backend_config = create_autospec(QasmBackendConfiguration)
    backend_config.max_shots = 1000

    backend = create_autospec(IBMBackend)
    backend.configuration.return_value = backend_config

    provider = create_autospec(IBMProvider)
    provider.get_backend.return_value = backend

    get_provider = Mock(return_value=provider)
    monkeypatch.setattr(
        "orquestra.integrations.qiskit.runner._ibmq_runner.get_provider", get_provider
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


def test_raises_on_unknown_backend_error(mock_execute: Mock, mock_ibm_backend: Mock):
    mock_execute.side_effect = IBMBackendApiError("unknown backend error")
    runner = create_ibmq_runner(
        api_token="mocked api token",
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1,
    )
    circuit = Circuit([H(0), CNOT(0, 1)])
    n_samples = 15
    with pytest.raises(RuntimeError):
        _ = runner.run_and_measure(circuit, n_samples)


def test_retry_on_too_many_jobs_error(
    mock_execute: Mock, mock_ibm_backend: Mock, monkeypatch: pytest.MonkeyPatch
):
    # Given
    # This mocks the response from IBM's API
    job_mock = create_autospec(IBMJob)
    result_mock = create_autospec(Result)
    result_mock.results = [Mock()]
    result_mock.get_memory.return_value = ["00", "01"]
    job_mock.result.return_value = result_mock
    mock_execute.side_effect = [
        IBMBackendApiError('"code":3458'),
        job_mock,
    ]

    # This prevents tests from needing to wait
    mock_sleep = Mock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    runner = create_ibmq_runner(
        api_token="mocked api token",
        backend_name="ibmq_qasm_simulator",
        retry_delay_seconds=1,
    )
    circuit = Circuit([H(0), CNOT(0, 1)])
    n_samples = 15

    # When
    _ = runner.run_and_measure(circuit, n_samples)

    # Then
    # This asserts that we had to wait because we had too many jobs running
    mock_sleep.assert_called()
