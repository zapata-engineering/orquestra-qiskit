# qe-qiskit

[![codecov](https://codecov.io/gh/zapatacomputing/qe-qiskit/branch/main/graph/badge.svg?token=G64YYS2IOS)](https://codecov.io/gh/zapatacomputing/orquestra-qiskit)

An Orquestra Resource for Qiskit

## Overview

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
For more details, see the [Orquestra Qiskit integration docs](http://docs.orquestra.io/other-resources/framework-integrations/qiskit/).

For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).
