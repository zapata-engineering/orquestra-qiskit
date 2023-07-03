################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
import qiskit
import qiskit.circuit.random
import sympy
from orquestra.quantum.circuits import (
    _builtin_gates,
    _circuit,
    _gates,
    _wavefunction_operations,
)

from orquestra.integrations.qiskit.conversions import (
    export_to_qiskit,
    import_from_qiskit,
)

# --------- gates ---------


EQUIVALENT_NON_PARAMETRIC_GATES = [
    (_builtin_gates.X, qiskit.circuit.library.XGate()),
    (_builtin_gates.Y, qiskit.circuit.library.YGate()),
    (_builtin_gates.Z, qiskit.circuit.library.ZGate()),
    (_builtin_gates.H, qiskit.circuit.library.HGate()),
    (_builtin_gates.I, qiskit.circuit.library.IGate()),
    (_builtin_gates.S, qiskit.circuit.library.SGate()),
    (_builtin_gates.SX, qiskit.circuit.library.SXGate()),
    (_builtin_gates.T, qiskit.circuit.library.TGate()),
    (_builtin_gates.CNOT, qiskit.extensions.CXGate()),
    (_builtin_gates.CZ, qiskit.extensions.CZGate()),
    (_builtin_gates.SWAP, qiskit.extensions.SwapGate()),
    (_builtin_gates.ISWAP, qiskit.extensions.iSwapGate()),
    (_builtin_gates.S.dagger, qiskit.extensions.SdgGate()),
    (_builtin_gates.T.dagger, qiskit.extensions.TdgGate()),
]

EQUIVALENT_PARAMETRIC_GATES = [
    (orquestra_cls(theta), qiskit_cls(theta))
    for orquestra_cls, qiskit_cls in [
        (_builtin_gates.RX, qiskit.circuit.library.RXGate),
        (_builtin_gates.RY, qiskit.circuit.library.RYGate),
        (_builtin_gates.RZ, qiskit.circuit.library.RZGate),
        (_builtin_gates.PHASE, qiskit.circuit.library.PhaseGate),
        (_builtin_gates.CPHASE, qiskit.extensions.CPhaseGate),
        (_builtin_gates.XX, qiskit.extensions.RXXGate),
        (_builtin_gates.YY, qiskit.extensions.RYYGate),
        (_builtin_gates.ZZ, qiskit.extensions.RZZGate),
    ]
    for theta in [0, -1, np.pi / 5, 2 * np.pi]
]


TWO_QUBIT_SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def _fix_qubit_ordering(qiskit_matrix):
    """Import qiskit matrix to Orquestra matrix convention.

    Qiskit uses different qubit ordering than we do.
    It causes multi-qubit matrices to look different on first sight."""
    if len(qiskit_matrix) == 2:
        return qiskit_matrix
    if len(qiskit_matrix) == 4:
        return TWO_QUBIT_SWAP_MATRIX @ qiskit_matrix @ TWO_QUBIT_SWAP_MATRIX
    else:
        raise ValueError(f"Unsupported matrix size: {len(qiskit_matrix)}")


class TestGateConversion:
    @pytest.mark.parametrize(
        "orquestra_gate,qiskit_gate",
        [
            *EQUIVALENT_NON_PARAMETRIC_GATES,
            *EQUIVALENT_PARAMETRIC_GATES,
        ],
    )
    def test_matrices_are_equal(self, orquestra_gate, qiskit_gate):
        orquestra_matrix = np.array(orquestra_gate.matrix).astype(np.complex128)
        qiskit_matrix = _fix_qubit_ordering(qiskit_gate.to_matrix())
        np.testing.assert_allclose(orquestra_matrix, qiskit_matrix)


class TestU3GateConversion:
    @pytest.mark.parametrize(
        "theta, phi, lambda_",
        [
            (0, 0, 0),
            (0, np.pi / 5, 0),
            (np.pi / 3, 0, 0),
            (0, 0, np.pi / 7),
            (42, -20, 30),
        ],
    )
    def test_matrices_are_equal_up_to_phase_factor(self, theta, phi, lambda_):
        orquestra_matrix = np.array(
            _builtin_gates.U3(theta, phi, lambda_).matrix
        ).astype(np.complex128)
        qiskit_matrix = qiskit.extensions.U3Gate(theta, phi, lambda_).to_matrix()

        np.testing.assert_allclose(orquestra_matrix, qiskit_matrix, atol=1e-7)


class TestCU3GateConversion:
    @pytest.mark.parametrize(
        "theta, phi, lambda_",
        [
            (0, 0, 0),
            (0, np.pi / 5, 0),
            (np.pi / 3, 0, 0),
            (0, 0, np.pi / 7),
            (42, -20, 30),
        ],
    )
    def test_matrices_are_equal_up_to_phase_factor(self, theta, phi, lambda_):
        orquestra_matrix = np.array(
            _builtin_gates.U3(theta, phi, lambda_).controlled(1)(0, 1).lifted_matrix(2)
        ).astype(np.complex128)
        qiskit_matrix = (
            qiskit.extensions.U3Gate(theta, phi, lambda_).control(1).to_matrix()
        )

        # Rearrange the qiskit matrix, such that it matches the endianness of orquestra
        qiskit_matrix_reversed_control = _fix_qubit_ordering(qiskit_matrix)

        np.testing.assert_allclose(
            orquestra_matrix, qiskit_matrix_reversed_control, atol=1e-7
        )


# --------- circuits ---------

# NOTE: In Qiskit, 0 is the most significant qubit,
# whereas in Orquestra, 0 is the least significant qubit.
# Thus, we need to flip the indices.
#
# See more at
# https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html#Visualize-Circuit


def _make_qiskit_circuit(n_qubits, commands, n_cbits=0):
    qc = qiskit.QuantumCircuit(n_qubits, n_cbits)
    for method_name, method_args in commands:
        method = getattr(qc, method_name)
        method(*method_args)
    return qc


SYMPY_THETA = sympy.Symbol("theta")
SYMPY_GAMMA = sympy.Symbol("gamma")
SYMPY_LAMBDA = sympy.Symbol("lambda_")
SYMPY_PARAMETER_VECTOR = [sympy.Symbol("p[0]"), sympy.Symbol("p[1]")]

QISKIT_THETA = qiskit.circuit.Parameter("theta")
QISKIT_GAMMA = qiskit.circuit.Parameter("gamma")
QISKIT_LAMBDA = qiskit.circuit.Parameter("lambda_")
QISKIT_PARAMETER_VECTOR = qiskit.circuit.ParameterVector("p", 2)


EXAMPLE_PARAM_VALUES = {
    "gamma": 0.3,
    "theta": -5,
    "lambda_": np.pi / 5,
    "p[0]": -5,
    "p[1]": 0.3,
}


EQUIVALENT_NON_PARAMETRIZED_CIRCUITS = [
    (
        _circuit.Circuit([], 3),
        _make_qiskit_circuit(3, []),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.X(0),
                _builtin_gates.Z(2),
            ],
            6,
        ),
        _make_qiskit_circuit(
            6,
            [
                ("x", (0,)),
                ("z", (2,)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.T.dagger(0),
            ],
            6,
        ),
        _make_qiskit_circuit(
            6,
            [
                ("tdg", (0,)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _wavefunction_operations.ResetOperation(0),
            ],
            6,
        ),
        _make_qiskit_circuit(
            6,
            [
                ("reset", (0,)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.X(0),
                _wavefunction_operations.ResetOperation(0),
            ],
            6,
        ),
        _make_qiskit_circuit(
            6,
            [
                ("x", (0,)),
                ("reset", (0,)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.CNOT(0, 1),
            ],
            4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("cnot", (0, 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(np.pi)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("rx", (np.pi, 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [_builtin_gates.SWAP.controlled(1)(2, 0, 3)],
            5,
        ),
        _make_qiskit_circuit(
            5,
            [
                ("append", (qiskit.circuit.library.SwapGate().control(1), [2, 0, 3])),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [_builtin_gates.Y.controlled(2)(4, 5, 2)],
            6,
        ),
        _make_qiskit_circuit(
            6,
            [
                ("append", (qiskit.circuit.library.YGate().control(2), [4, 5, 2])),
            ],
        ),
    ),
    (
        _circuit.Circuit([_builtin_gates.U3(np.pi / 5, np.pi / 2, np.pi / 4)(2)]),
        _make_qiskit_circuit(
            3,
            [
                (
                    "append",
                    (qiskit.extensions.U3Gate(np.pi / 5, np.pi / 2, np.pi / 4), [2]),
                )
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [_builtin_gates.U3(np.pi / 5, np.pi / 2, np.pi / 4).controlled(1)(1, 2)]
        ),
        _make_qiskit_circuit(
            3,
            [
                (
                    "append",
                    (
                        qiskit.extensions.U3Gate(
                            np.pi / 5, np.pi / 2, np.pi / 4
                        ).control(1),
                        [1, 2],
                    ),
                )
            ],
        ),
    ),
    (
        _circuit.Circuit([_builtin_gates.Delay(1)(0)]),
        _make_qiskit_circuit(1, [("delay", (1, 0))]),
    ),
]


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("rx", (QISKIT_THETA, 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA * SYMPY_GAMMA)(1),
            ],
            4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("rx", (QISKIT_THETA * QISKIT_GAMMA, 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [_builtin_gates.U3(SYMPY_THETA, SYMPY_GAMMA, SYMPY_LAMBDA)(3)]
        ),
        _make_qiskit_circuit(
            4,
            [
                (
                    "append",
                    (
                        qiskit.extensions.U3Gate(
                            QISKIT_THETA, QISKIT_GAMMA, QISKIT_LAMBDA
                        ),
                        [3],
                    ),
                )
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.U3(SYMPY_THETA, SYMPY_GAMMA, SYMPY_LAMBDA).controlled(1)(
                    2, 3
                )
            ]
        ),
        _make_qiskit_circuit(
            4,
            [
                (
                    "append",
                    (
                        qiskit.extensions.U3Gate(
                            QISKIT_THETA, QISKIT_GAMMA, QISKIT_LAMBDA
                        ).control(1),
                        [2, 3],
                    ),
                )
            ],
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(
                    SYMPY_PARAMETER_VECTOR[0] * SYMPY_PARAMETER_VECTOR[1]
                )(1),
            ],
            4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("rx", (QISKIT_PARAMETER_VECTOR[0] * QISKIT_PARAMETER_VECTOR[1], 1)),
            ],
        ),
    ),
]


UNITARY_GATE_DEF = _gates.CustomGateDefinition(
    "unitary.33c11b461fe67e717e37ac34a568cd1c27a89013703bf5b84194f0732a33a26d",
    sympy.Matrix([[0, 1], [1, 0]]),
    tuple(),
)
CUSTOM_A2_GATE_DEF = _gates.CustomGateDefinition(
    "custom.A2.33c11b461fe67e717e37ac34a568cd1c27a89013703bf5b84194f0732a33a26d",
    sympy.Matrix([[0, 1], [1, 0]]),
    tuple(),
)


EQUIVALENT_CUSTOM_GATE_CIRCUITS = [
    (
        _circuit.Circuit(
            operations=[UNITARY_GATE_DEF()(1)],
            n_qubits=4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("unitary", (np.array([[0, 1], [1, 0]]), 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            operations=[CUSTOM_A2_GATE_DEF()(3)],
            n_qubits=5,
        ),
        _make_qiskit_circuit(
            5,
            [
                ("unitary", (np.array([[0, 1], [1, 0]]), 3, "custom.A2")),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            operations=[UNITARY_GATE_DEF()(1), UNITARY_GATE_DEF()(1)],
            n_qubits=4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("unitary", (np.array([[0, 1], [1, 0]]), 1)),
                ("unitary", (np.array([[0, 1], [1, 0]]), 1)),
            ],
        ),
    ),
    (
        _circuit.Circuit(
            operations=[
                UNITARY_GATE_DEF()(1),
                CUSTOM_A2_GATE_DEF()(1),
                UNITARY_GATE_DEF()(0),
            ],
            n_qubits=4,
        ),
        _make_qiskit_circuit(
            4,
            [
                ("unitary", (np.array([[0, 1], [1, 0]]), 1)),
                ("unitary", (np.array([[0, 1], [1, 0]]), 1, "custom.A2")),
                ("unitary", (np.array([[0, 1], [1, 0]]), 0)),
            ],
        ),
    ),
]

UNSUPPORTED_CIRCUITS = [
    _make_qiskit_circuit(1, [("measure", (0, 0))], n_cbits=1),
    _make_qiskit_circuit(1, [("break_loop", ())]),
]


def _draw_qiskit_circuit(circuit):
    return qiskit.visualization.circuit_drawer(circuit, output="text")


class TestExportingToQiskit:
    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_NON_PARAMETRIZED_CIRCUITS
    )
    def test_exporting_circuit_gives_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        converted = export_to_qiskit(orquestra_circuit)
        assert converted == qiskit_circuit, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted)}\n isn't equal "
            f"to\n{_draw_qiskit_circuit(qiskit_circuit)}"
        )

    @pytest.mark.parametrize(
        "orquestra_circuit",
        [
            orquestra_circuit
            for orquestra_circuit, _ in EQUIVALENT_PARAMETRIZED_CIRCUITS
        ],
    )
    def test_exporting_parametrized_circuit_doesnt_change_symbol_names(
        self, orquestra_circuit
    ):
        converted = export_to_qiskit(orquestra_circuit)
        converted_names = sorted(map(str, converted.parameters))
        initial_names = sorted(map(str, orquestra_circuit.free_symbols))
        assert converted_names == initial_names

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_exporting_and_binding_parametrized_circuit_results_in_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        # 1. Export
        converted = export_to_qiskit(orquestra_circuit)
        # 2. Bind params
        converted_bound = converted.bind_parameters(
            {param: EXAMPLE_PARAM_VALUES[str(param)] for param in converted.parameters}
        )

        # 3. Bind the ref
        ref_bound = qiskit_circuit.bind_parameters(
            {
                param: EXAMPLE_PARAM_VALUES[str(param)]
                for param in qiskit_circuit.parameters
            }
        )
        assert converted_bound == ref_bound, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted_bound)}\n isn't "
            f"equal to\n{_draw_qiskit_circuit(ref_bound)}"
        )

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_binding_and_exporting_parametrized_circuit_results_in_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        # 1. Bind params
        bound = orquestra_circuit.bind(
            {
                symbol: EXAMPLE_PARAM_VALUES[str(symbol)]
                for symbol in orquestra_circuit.free_symbols
            }
        )
        # 2. Export
        bound_converted = export_to_qiskit(bound)

        # 3. Bind the ref
        ref_bound = qiskit_circuit.bind_parameters(
            {
                param: EXAMPLE_PARAM_VALUES[str(param)]
                for param in qiskit_circuit.parameters
            }
        )
        assert bound_converted == ref_bound, (
            f"Converted circuit:\n{_draw_qiskit_circuit(bound_converted)}\n isn't "
            f"equal to\n{_draw_qiskit_circuit(ref_bound)}"
        )

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_CUSTOM_GATE_CIRCUITS
    )
    def test_exporting_circuit_with_custom_gates_gives_equivalent_operator(
        self, orquestra_circuit, qiskit_circuit
    ):
        exported = export_to_qiskit(orquestra_circuit)
        # We can't compare the circuits directly, because the gate names can differ.
        # Qiskit allows multiple gate operations with the same label. Orquestra doesn't
        # allow that, so we append a matrix hash to the name.
        assert qiskit.quantum_info.Operator(exported) == qiskit.quantum_info.Operator(
            qiskit_circuit
        )


class TestImportingFromQiskit:
    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_NON_PARAMETRIZED_CIRCUITS
    )
    def test_importing_circuit_gives_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        imported = import_from_qiskit(qiskit_circuit)
        assert imported == orquestra_circuit

    @pytest.mark.parametrize(
        "qiskit_circuit",
        [q_circuit for _, q_circuit in EQUIVALENT_PARAMETRIZED_CIRCUITS],
    )
    def test_importing_parametrized_circuit_doesnt_change_symbol_names(
        self, qiskit_circuit
    ):
        imported = import_from_qiskit(qiskit_circuit)
        assert sorted(map(str, imported.free_symbols)) == sorted(
            map(str, qiskit_circuit.parameters)
        )

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_importing_and_binding_parametrized_circuit_results_in_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        # 1. Import
        imported = import_from_qiskit(qiskit_circuit)
        symbols_map = {
            symbol: EXAMPLE_PARAM_VALUES[str(symbol)]
            for symbol in imported.free_symbols
        }
        # 2. Bind params
        imported_bound = imported.bind(symbols_map)

        # 3. Bind the ref
        ref_bound = orquestra_circuit.bind(symbols_map)

        assert imported_bound == ref_bound

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_binding_and_importing_parametrized_circuit_results_in_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        # 1. Bind params
        bound = qiskit_circuit.bind_parameters(
            {
                param: EXAMPLE_PARAM_VALUES[str(param)]
                for param in qiskit_circuit.parameters
            }
        )
        # 2. Import
        bound_imported = import_from_qiskit(bound)

        # 3. Bind the ref
        ref_bound = orquestra_circuit.bind(
            {
                symbol: EXAMPLE_PARAM_VALUES[str(symbol)]
                for symbol in orquestra_circuit.free_symbols
            }
        )
        assert bound_imported == ref_bound

    @pytest.mark.parametrize(
        "orquestra_circuit, qiskit_circuit", EQUIVALENT_CUSTOM_GATE_CIRCUITS
    )
    def test_importing_circuit_with_custom_gates_gives_equivalent_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        imported = import_from_qiskit(qiskit_circuit)
        assert imported == orquestra_circuit

    @pytest.mark.parametrize("unsupported_circuit", UNSUPPORTED_CIRCUITS)
    def test_operation_not_implemented(self, unsupported_circuit):
        with pytest.raises(ValueError):
            import_from_qiskit(unsupported_circuit)
