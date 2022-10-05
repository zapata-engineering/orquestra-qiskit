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
from qiskit.opflow import PauliOp, SummedOp
from qiskit.quantum_info import Pauli


def qubitop_to_qiskitpauli(operator: PauliRepresentation) -> SummedOp:
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
        terms.append(PauliOp(Pauli(string_term), coeff=term.coefficient))

    return SummedOp(terms)


def qiskitpauli_to_qubitop(qiskit_pauli: SummedOp) -> PauliSum:
    """Convert a qiskit's SummedOp to a PauliSum.

    Args:
        qiskit_pauli: operator to convert

    Returns:
        PauliSum representing the SummedOp
    """

    if not isinstance(qiskit_pauli, SummedOp):
        raise TypeError("qiskit_pauli must be a qiskit SummedOp")

    transformed_operator = PauliSum()

    for pauli_op in qiskit_pauli._oplist:
        qiskit_term, weight = pauli_op.primitive, pauli_op.coeff

        orquestra_term = PauliTerm.identity()
        for (term_qubit, term_pauli) in enumerate(str(qiskit_term)):
            if term_pauli != "I":
                if orquestra_term == PauliTerm.identity():
                    orquestra_term = PauliTerm(f"{term_pauli}{term_qubit}")
                else:
                    product = PauliTerm(f"{term_pauli}{term_qubit}") * orquestra_term
                    assert isinstance(product, PauliTerm)
                    orquestra_term = product

        transformed_operator += orquestra_term * weight

    return transformed_operator
