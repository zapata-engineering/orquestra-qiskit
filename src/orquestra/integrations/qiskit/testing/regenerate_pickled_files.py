"""Every time the Qiskit version is updated, the pickled files need to be regenerated.
This script regenerates the pickled files."""

import os
import pickle

from orquestra.quantum.circuits import CNOT, Circuit, X

from orquestra.integrations.qiskit.backend import QiskitBackend

token = "YOUR_TOKEN_HERE"
file_path = os.path.dirname(os.path.realpath(__file__))


def pickle_jobs_and_batches_with_different_qubits():
    """Create pickle file for jobs and batches with different qubits."""

    # make orquestra circuit
    circuits = [Circuit([X(0), CNOT(0, 3), CNOT(1, 2)])]
    n_samples = 50
    lima_backend = QiskitBackend("ibmq_lima", api_token=token)

    (
        experiments,
        n_samples_for_experiments,
        multiplicities,
    ) = lima_backend.transform_circuitset_to_ibmq_experiments(circuits, [n_samples])
    (
        batches,
        n_samples_for_batches,
    ) = lima_backend.batch_experiments(experiments, n_samples_for_experiments)

    jobs = [
        lima_backend.execute_with_retries(batch, n_samples)
        for n_samples, batch in zip(n_samples_for_batches, batches)
    ]
    for job in jobs:
        job.result()

    # we dont care about the results, we just want to pickle the jobs object
    jobs[0].result().results[0].data.counts["0x9"] = n_samples

    with open(
        os.path.join(file_path, "jobs_and_batches_with_different_qubits.pickle"), "wb"
    ) as f:
        pickle.dump(jobs, f)
        pickle.dump(batches, f)
        pickle.dump(multiplicities, f)


if __name__ == "__main__":
    pickle_jobs_and_batches_with_different_qubits()
