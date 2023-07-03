################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import hashlib
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple, Union

import numpy as np
import qiskit
import sympy
from orquestra.quantum.circuits import (
    _builtin_gates,
    _circuit,
    _gates,
    _operations,
    _wavefunction_operations,
)
from orquestra.quantum.circuits.symbolic.sympy_expressions import (
    SYMPY_DIALECT,
    expression_from_sympy,
)
from orquestra.quantum.circuits.symbolic.translations import translate_expression

from ._qiskit_expressions import QISKIT_DIALECT, expression_from_qiskit

QiskitTriplet = Tuple[
    qiskit.circuit.Instruction, List[qiskit.circuit.Qubit], List[qiskit.circuit.Clbit]
]


def _import_qiskit_qubit(qubit: qiskit.circuit.Qubit) -> int:
    return qubit.index


def _qiskit_expr_from_orquestra(expr):
    intermediate = expression_from_sympy(expr)
    return translate_expression(intermediate, QISKIT_DIALECT)


def _orquestra_expr_from_qiskit(expr):
    intermediate = expression_from_qiskit(expr)
    return translate_expression(intermediate, SYMPY_DIALECT)


ORQUESTRA_QISKIT_GATE_MAP = {
    _builtin_gates.X: qiskit.circuit.library.XGate,
    _builtin_gates.Y: qiskit.circuit.library.YGate,
    _builtin_gates.Z: qiskit.circuit.library.ZGate,
    _builtin_gates.S: qiskit.circuit.library.SGate,
    _builtin_gates.SX: qiskit.circuit.library.SXGate,
    _builtin_gates.T: qiskit.circuit.library.TGate,
    _builtin_gates.H: qiskit.circuit.library.HGate,
    _builtin_gates.I: qiskit.circuit.library.IGate,
    _builtin_gates.CNOT: qiskit.circuit.library.CXGate,
    _builtin_gates.CZ: qiskit.circuit.library.CZGate,
    _builtin_gates.SWAP: qiskit.circuit.library.SwapGate,
    _builtin_gates.ISWAP: qiskit.circuit.library.iSwapGate,
    _builtin_gates.RX: qiskit.circuit.library.RXGate,
    _builtin_gates.RY: qiskit.circuit.library.RYGate,
    _builtin_gates.RZ: qiskit.circuit.library.RZGate,
    _builtin_gates.PHASE: qiskit.circuit.library.PhaseGate,
    _builtin_gates.CPHASE: qiskit.circuit.library.CPhaseGate,
    _builtin_gates.XX: qiskit.circuit.library.RXXGate,
    _builtin_gates.YY: qiskit.circuit.library.RYYGate,
    _builtin_gates.ZZ: qiskit.circuit.library.RZZGate,
    _builtin_gates.U3: qiskit.circuit.library.U3Gate,
    _builtin_gates.Delay: qiskit.circuit.Delay,
}


def _make_gate_instance(gate_ref, gate_params) -> _gates.Gate:
    """Returns a gate instance that's applicable to qubits.
    For non-parametric gate refs like X, returns just the `X`
    For parametric gate factories like `RX`, returns the produced gate, like `RX(0.2)`
    """
    if _gates.gate_is_parametric(gate_ref, gate_params):
        return gate_ref(*gate_params)
    else:
        return gate_ref


def _make_controlled_gate_prototype(wrapped_gate_ref, num_control_qubits=1):
    def _factory(*gate_params):
        return _gates.ControlledGate(
            _make_gate_instance(wrapped_gate_ref, gate_params), num_control_qubits
        )

    return _factory


QISKIT_ORQUESTRA_GATE_MAP = {
    **{q_cls: z_ref for z_ref, q_cls in ORQUESTRA_QISKIT_GATE_MAP.items()},
    qiskit.extensions.SXdgGate: _builtin_gates.SX.dagger,
    qiskit.extensions.SdgGate: _builtin_gates.S.dagger,
    qiskit.extensions.TdgGate: _builtin_gates.T.dagger,
    qiskit.circuit.library.CSwapGate: _builtin_gates.SWAP.controlled(1),
    qiskit.circuit.library.CRXGate: _make_controlled_gate_prototype(_builtin_gates.RX),
    qiskit.circuit.library.CRYGate: _make_controlled_gate_prototype(_builtin_gates.RY),
    qiskit.circuit.library.CRZGate: _make_controlled_gate_prototype(_builtin_gates.RZ),
}


def export_to_qiskit(circuit: _circuit.Circuit) -> qiskit.QuantumCircuit:
    q_circuit = qiskit.QuantumCircuit(circuit.n_qubits)
    q_register = qiskit.circuit.QuantumRegister(circuit.n_qubits, "q")
    gate_op_only_circuit = _circuit.Circuit(
        [op for op in circuit.operations if isinstance(op, _gates.GateOperation)]
    )
    custom_names = {
        gate_def.gate_name
        for gate_def in gate_op_only_circuit.collect_custom_gate_definitions()
    }
    q_triplets = []
    for gate_op in circuit.operations:
        if isinstance(gate_op, _gates.GateOperation):
            q_triplet = _export_gate_to_qiskit(
                gate_op.gate,
                applied_qubit_indices=gate_op.qubit_indices,
                q_register=q_register,
                custom_names=custom_names,
            )
        elif isinstance(gate_op, _wavefunction_operations.ResetOperation):
            q_triplet = (
                qiskit.circuit.library.Reset(),
                [q_register[gate_op.qubit_indices[0]]],
                [],
            )
        q_triplets.append(q_triplet)

    for q_gate, q_qubits, q_clbits in q_triplets:
        q_circuit.append(q_gate, q_qubits, q_clbits)
    return q_circuit


def _export_gate_to_qiskit(gate, applied_qubit_indices, q_register, custom_names):
    try:
        return _export_gate_via_mapping(
            gate, applied_qubit_indices, q_register, custom_names
        )
    except ValueError:
        pass

    try:
        return _export_dagger_gate(
            gate, applied_qubit_indices, q_register, custom_names
        )
    except ValueError:
        pass

    try:
        return _export_controlled_gate(
            gate, applied_qubit_indices, q_register, custom_names
        )
    except ValueError:
        pass

    try:
        return _export_custom_gate(
            gate, applied_qubit_indices, q_register, custom_names
        )
    except ValueError:
        pass

    raise NotImplementedError(f"Exporting gate {gate} to Qiskit is unsupported")


def _export_gate_via_mapping(gate, applied_qubit_indices, q_register, custom_names):
    try:
        qiskit_cls = ORQUESTRA_QISKIT_GATE_MAP[
            _builtin_gates.builtin_gate_by_name(gate.name)
        ]
    except KeyError:
        raise ValueError(f"Can't export gate {gate} to Qiskit via mapping")

    qiskit_params = [_qiskit_expr_from_orquestra(param) for param in gate.params]

    qiskit_qubits = [q_register[index] for index in applied_qubit_indices]

    return qiskit_cls(*qiskit_params), qiskit_qubits, []


def _export_dagger_gate(
    gate: _gates.Dagger,
    applied_qubit_indices,
    q_register,
    custom_names,
):
    if not isinstance(gate, _gates.Dagger):
        # Raising an exception here is redundant to the type hint, but it allows us
        # to handle exporting all gates in the same way, regardless of type
        raise ValueError(f"Can't export gate {gate} as a dagger gate")

    target_gate, qiskit_qubits, qiskit_clbits = _export_gate_to_qiskit(
        gate.wrapped_gate,
        applied_qubit_indices=applied_qubit_indices,
        q_register=q_register,
        custom_names=custom_names,
    )
    return target_gate.inverse(), qiskit_qubits, qiskit_clbits


def _export_controlled_gate(
    gate: _gates.ControlledGate,
    applied_qubit_indices,
    q_register,
    custom_names,
):
    if not isinstance(gate, _gates.ControlledGate):
        # Raising an exception here is redundant to the type hint, but it allows us
        # to handle exporting all gates in the same way, regardless of type
        raise ValueError(f"Can't export gate {gate} as a controlled gate")

    target_indices = applied_qubit_indices[gate.num_control_qubits :]
    target_gate, _, _ = _export_gate_to_qiskit(
        gate.wrapped_gate,
        applied_qubit_indices=target_indices,
        q_register=q_register,
        custom_names=custom_names,
    )
    controlled_gate = target_gate.control(gate.num_control_qubits)

    qiskit_qubits = [q_register[index] for index in applied_qubit_indices]

    return controlled_gate, qiskit_qubits, []


def _export_custom_gate(
    gate: _gates.MatrixFactoryGate,
    applied_qubit_indices,
    q_register,
    custom_names,
):
    if gate.name not in custom_names:
        raise ValueError(
            f"Can't export gate {gate} as a custom gate, the circuit is missing its "
            "definition"
        )

    if gate.params:
        raise ValueError(
            f"Can't export parametrized gate {gate}, Qiskit doesn't support "
            "parametrized custom gates"
        )
    # At that time of writing it Qiskit doesn't support parametrized gates defined with
    # a symbolic matrix.
    # See https://github.com/Qiskit/qiskit-terra/issues/4751 for more info.

    qiskit_qubits = [q_register[index] for index in applied_qubit_indices]

    qiskit_matrix = np.array(gate.matrix)
    return (
        qiskit.extensions.UnitaryGate(qiskit_matrix, label=gate.name),
        qiskit_qubits,
        [],
    )


class AnonGateOperation(NamedTuple):
    gate_name: str
    matrix: sympy.Matrix
    qubit_indices: Tuple[int, ...]


ImportedOperation = Union[_operations.Operation, AnonGateOperation]


def _apply_custom_gate(
    anon_op: AnonGateOperation, custom_defs_map: Dict[str, _gates.CustomGateDefinition]
) -> _gates.GateOperation:
    gate_def = custom_defs_map[anon_op.gate_name]
    # Qiskit doesn't support custom gates with parametrized matrices
    # so we can assume empty params list.
    gate_params: Tuple[sympy.Symbol, ...] = tuple()
    gate = gate_def(*gate_params)

    return gate(*anon_op.qubit_indices)


def import_from_qiskit(circuit: qiskit.QuantumCircuit) -> _circuit.Circuit:
    q_ops = [_import_qiskit_triplet(triplet) for triplet in circuit.data]
    anon_ops = [op for op in q_ops if isinstance(op, AnonGateOperation)]

    # Qiskit doesn't support custom gates with parametrized matrices
    # so we can assume empty params list.
    params_ordering: Tuple[sympy.Symbol, ...] = tuple()
    custom_defs = {
        anon_op.gate_name: _gates.CustomGateDefinition(
            gate_name=anon_op.gate_name,
            matrix=anon_op.matrix,
            params_ordering=params_ordering,
        )
        for anon_op in anon_ops
    }
    imported_ops = [
        _apply_custom_gate(op, custom_defs) if isinstance(op, AnonGateOperation) else op
        for op in q_ops
    ]
    return _circuit.Circuit(
        operations=imported_ops,
        n_qubits=circuit.num_qubits,
    )


def _import_qiskit_triplet(qiskit_triplet: QiskitTriplet) -> ImportedOperation:
    qiskit_op, qiskit_qubits, _ = qiskit_triplet

    return _import_qiskit_op(qiskit_op, qiskit_qubits)


def _import_qiskit_op(qiskit_op, qiskit_qubits) -> ImportedOperation:
    # We always wanna try importing via mapping to handle complex gate structures
    # represented by a single class, like CNOT (Control + X) or CSwap (Control + Swap).
    try:
        return _import_qiskit_op_via_mapping(qiskit_op, qiskit_qubits)
    except ValueError:
        pass

    try:
        return _import_controlled_qiskit_op(qiskit_op, qiskit_qubits)
    except ValueError:
        pass

    try:
        return _import_custom_qiskit_gate(qiskit_op, qiskit_qubits)
    except AttributeError:
        raise ValueError(f"Conversion of {qiskit_op.name} from Qiskit is unsupported.")


def _import_qiskit_op_via_mapping(
    qiskit_gate: qiskit.circuit.Instruction,
    qiskit_qubits: Iterable[qiskit.circuit.Qubit],
) -> _operations.Operation:
    qubit_indices = [_import_qiskit_qubit(qubit) for qubit in qiskit_qubits]

    if isinstance(qiskit_gate, qiskit.circuit.library.Reset):
        return _wavefunction_operations.ResetOperation(qubit_indices[0])

    try:
        gate_ref = QISKIT_ORQUESTRA_GATE_MAP[type(qiskit_gate)]
    except KeyError:
        raise ValueError(f"Conversion of {qiskit_gate} from Qiskit is unsupported.")

    # values to consider:
    # - gate matrix parameters (only parametric gates)
    # - gate application indices (all gates)
    orquestra_params = [
        _orquestra_expr_from_qiskit(param) for param in qiskit_gate.params
    ]
    gate = _make_gate_instance(gate_ref, orquestra_params)
    return _gates.GateOperation(gate=gate, qubit_indices=tuple(qubit_indices))


def _import_controlled_qiskit_op(
    qiskit_gate: qiskit.circuit.ControlledGate,
    qiskit_qubits: Sequence[qiskit.circuit.Qubit],
) -> _gates.GateOperation:
    if not isinstance(qiskit_gate, qiskit.circuit.ControlledGate):
        # Raising an exception here is redundant to the type hint, but it allows us
        # to handle exporting all gates in the same way, regardless of type
        raise ValueError(f"Can't import gate {qiskit_gate} as a controlled gate")

    wrapped_qubits = qiskit_qubits[qiskit_gate.num_ctrl_qubits :]
    wrapped_op = _import_qiskit_op(qiskit_gate.base_gate, wrapped_qubits)
    qubit_indices = map(_import_qiskit_qubit, qiskit_qubits)
    if isinstance(wrapped_op, _gates.GateOperation):
        return wrapped_op.gate.controlled(qiskit_gate.num_ctrl_qubits)(*qubit_indices)
    else:
        raise NotImplementedError(
            "Importing of controlled anonymous gates not yet supported."
        )


def _hash_hex(bytes_):
    return hashlib.sha256(bytes_).hexdigest()


def _custom_qiskit_gate_name(gate_label: str, gate_name: str, matrix: np.ndarray):
    matrix_hash = _hash_hex(matrix.tobytes())
    target_name = gate_label or gate_name
    return f"{target_name}.{matrix_hash}"


def _import_custom_qiskit_gate(
    qiskit_op: qiskit.circuit.Gate, qiskit_qubits: Iterable[qiskit.circuit.Qubit]
) -> AnonGateOperation:
    value_matrix = qiskit_op.to_matrix()
    return AnonGateOperation(
        gate_name=_custom_qiskit_gate_name(
            qiskit_op.label, qiskit_op.name, value_matrix
        ),
        matrix=sympy.Matrix(value_matrix),
        qubit_indices=tuple(_import_qiskit_qubit(qubit) for qubit in qiskit_qubits),
    )
