# orquestra-qiskit

## What is it?

`orquestra-qiskit` is a [Zapata](https://www.zapatacomputing.com) library holding modules for integrating qiskit with [Orquestra](https://www.zapatacomputing.com/orquestra/).

## Installation

Even though it's intended to be used with Orquestra, `orquestra-qiskit` can be also used as a Python module.
Just run `pip install .` from the main directory.

## Usage

`orquestra-qiskit` is a Python module that exposes Qiskit's simulators as an [`orquestra`](https://github.com/zapatacomputing/orquestra-quantum/blob/main/src/orquestra/quantum/api/backend.py) `QuantumSimulator`. It can be imported with:

```
from orquestra.integrations.qiskit.simulator import QiskitSimulator
```

It also exposes Qiskit's quantum backends as a `QiskitBackend` which implements the `orquestra.quantum.api.backend.QuantumBackend` interface.

It can be imported with:

```
from orquestra.integrations.qiskit.backend import QiskitBackend
```

In addition, it also provides converters that allow switching between `qiskit` circuits and those of `orquestra`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Core docs](https://zapatacomputing.github.io/orquestra-core/index.html).

For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).
