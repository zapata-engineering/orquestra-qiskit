############################################################################
#   Copyright 2017 Rigetti Computing, Inc.
#   Modified by Zapata Computing 2020 to work for qiskit's SummedOp.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
############################################################################

"""
Translates OpenFermion Objects to qiskit SummedOp objects
"""
from orquestra.quantum.operators import PauliRepresentation, PauliSum, PauliTerm
from qiskit.quantum_info import SparsePauliOp


def qubitop_to_qiskitpauli(operator: PauliRepresentation) -> SparsePauliOp:
    """Convert a PauliRepresentation to a SummedOp.

    Args:
        operator: PauliRepresentation to convert

    Returns:
        SummedOp representing the qubit operator
    """
    if not isinstance(operator, PauliSum) and not isinstance(operator, PauliTerm):
        raise TypeError("operator must be an Orquestra PauliSum or PauliTerm object")

    terms = []
    for term in operator.terms:
        string_term = "I" * len(operator)
        for term_qubit, term_pauli in term.operations:
            string_term = (
                string_term[:term_qubit] + term_pauli + string_term[term_qubit + 1 :]
            )
        terms.append((string_term, term.coefficient))

    if not terms:
        return SparsePauliOp("")
    else:
        return SparsePauliOp.from_list(terms)


def qiskitpauli_to_qubitop(qiskit_pauli: SparsePauliOp) -> PauliSum:
    """Convert a qiskit's SummedOp to a PauliSum.

    Args:
        qiskit_pauli: operator to convert

    Returns:
        PauliSum representing the SummedOp
    """
    if not isinstance(qiskit_pauli, SparsePauliOp):
        raise TypeError("qiskit_pauli must be SparsePauliOp object")

    transformed_operator = PauliSum()

    for qiskit_term, weight in zip(qiskit_pauli.paulis, qiskit_pauli.coeffs):
        orquestra_term = PauliTerm.identity()
        for term_qubit, term_pauli in enumerate(str(qiskit_term)):
            if term_pauli != "I":
                if orquestra_term == PauliTerm.identity():
                    orquestra_term = PauliTerm(f"{term_pauli}{term_qubit}")
                else:
                    product = PauliTerm(f"{term_pauli}{term_qubit}") * orquestra_term
                    assert isinstance(product, PauliTerm)
                    orquestra_term = product

        transformed_operator += orquestra_term * weight

    return transformed_operator
