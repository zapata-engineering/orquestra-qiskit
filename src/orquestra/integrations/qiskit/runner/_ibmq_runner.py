import sys
import time
from typing import List, Optional

from qiskit import execute
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMBackendApiError

from .._get_provider import get_provider
from ._qiskit_runner import QiskitRunner


def _execute_on_ibmq_with_retries(
    retry_delay_seconds: int, retry_timeout_seconds: int = sys.maxsize
):
    def _execute(*args, **kwargs):
        start_time = time.time()
        while (elapsed_seconds := (time.time() - start_time)) < retry_timeout_seconds:
            try:
                return execute(*args, **kwargs)
            except IBMBackendApiError as e:
                # 3458 is the error code for too many jobs
                # https://docs.quantum-computing.ibm.com/errors
                if '"code":3458' not in str(e):
                    raise RuntimeError(e) from e
                print(f"Job limit reached. Retrying in {retry_delay_seconds}s.")
                time.sleep(retry_delay_seconds)
        raise RuntimeError(
            f"Failed to submit job in {elapsed_seconds}s due to backend job " "limit."
        )

    return _execute


def create_ibmq_runner(
    api_token: str,
    backend_name: str,
    hub: str = "ibm-q",
    group: str = "open",
    project: str = "main",
    noise_model: Optional[NoiseModel] = None,
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 0,
    seed: Optional[int] = None,
    retry_delay_seconds: int = 60,
    retry_timeout_seconds: int = 24 * 60 * 60,  # default timeout of one day
    discard_extra_measurements=False,
):
    provider = get_provider(api_token=api_token, hub=hub, group=group, project=project)
    backend = provider.get_backend(name=backend_name)

    return QiskitRunner(
        backend,
        noise_model=noise_model,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed=seed,
        execute_function=_execute_on_ibmq_with_retries(
            retry_delay_seconds, retry_timeout_seconds
        ),
        discard_extra_measurements=discard_extra_measurements,
    )
