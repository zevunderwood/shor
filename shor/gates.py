import math
from typing import Iterable, Union

import numpy as np

from shor.errors import CircuitError
from shor.layers import _Layer
from shor.utils.collections import flatten

QbitOrIterable = Union[int, Iterable]


class _Gate(_Layer):
    """Abstract base quantum gate class

    # Properties
    input_length = valid length of input qubits
    qubits = indices of qubits, to be used as input to gate.
    """

    @property
    def symbol(self):
        return self.__class__.__name__.lower()

    def __init__(self, bool=None, cbit=None, *qbits: QbitOrIterable, **kwargs):
        super().__init__(**kwargs)
        self.qbits = flatten(qbits) if qbits else [0]
        self.dimension = kwargs.get("dimension", 1)
        self.bool = bool
        self.cbit = cbit

        assert all(map(lambda q: type(q) == int, self.qbits)), str(self.qbits)
        try:
            assert len(self.qbits) % self.dimension == 0
        except AssertionError:
            raise CircuitError(
                f"The input qbits length {len(self.qbits)} is not divisible by the '{self.symbol}' "
                f"gate's dimension {self.dimension}"
            )

    @property
    def qubits(self):
        return self.qbits

    def to_gates(self):
        if len(self.qbits) > self.dimension:
            return [
                self.__class__(self.qbits[i : i + self.dimension]) for i in range(0, len(self.qbits), self.dimension)
            ]
        return [self]

    @property
    def num_states(self):
        return np.power(2, self.dimension)

    def to_matrix(self) -> np.ndarray:
        return np.eye(self.num_states())

    @property
    def matrix(self):
        return self.to_matrix()

    def invert(self):
        return self

    @property
    def I(self):
        return self.invert()

    def __invert__(self):
        return self.invert()

    def apply_if(self, cbit, bool):
        self.bool = bool
        self.cbit = cbit
        # Send this over to __init__? now to finish initializing the gate


class CNOT(_Gate):
    symbol = "CX"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class CY(_Gate):
    symbol = "CY"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1 * 1j], [0, 0, 1j, 0]])


class CSWAP(_Gate):
    symbol = "CSWAP"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 3
        if not qubits:
            qubits = [0, 1, 2]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        cswap_matrix = np.eye(8)
        cswap_matrix[:, [5, 6]] = cswap_matrix[:, [6, 5]]
        return cswap_matrix


class Hadamard(_Gate):
    symbol = "H"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.multiply(np.divide(1, np.sqrt(self.num_states)), np.array([[1, 1], [1, -1]]))


class PauliX(_Gate):
    symbol = "X"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[0, 1], [1, 0]])


class PauliY(_Gate):
    symbol = "Y"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])


class PauliZ(_Gate):
    symbol = "Z"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, -1]])


class QFT(_Gate):
    def __init__(self, *qubits, **kwargs):
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, dimension=len(qubits), **kwargs)

    # def to_gates(self):
    #     # TODO: translate this gate to base gates / CNOTs
    #     pass

    def get_nth_unity_root(self, k):
        return np.exp((2j * np.pi * k) / self.num_states)

    def to_matrix(self) -> np.ndarray:
        m = np.array(np.ones((self.num_states, self.num_states)), dtype="complex")

        for i in range(1, self.num_states):
            for j in range(i, self.num_states):
                w = self.get_nth_unity_root(i * j)
                m[i, j] = w
                m[j, i] = w

        return np.around(np.multiply(1 / np.sqrt(self.num_states), m), decimals=15)


class SWAP(_Gate):
    symbol = "SWAP"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class Cx(_Gate):
    symbol = "CX"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


class CCNOT(_Gate):
    symbol = "CCX"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 3
        if not qubits:
            qubits = [0, 1, 2]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )


class CRZ(_Gate):
    symbol = "CRZ"

    def __init__(self, *qubits, angle=0, **kwargs):
        kwargs["dimension"] = 2
        self.angle = angle
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * self.angle / 2), 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ]
        )


class CH(_Gate):
    symbol = "CH"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )


class S(_Gate):
    symbol = "S"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, 1j]])


class Sdg(_Gate):
    symbol = "Sdg"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, -1j]])


class T(_Gate):
    symbol = "T"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class Tdg(_Gate):
    symbol = "Tdg"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])


class ID(_Gate):
    symbol = "I"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0], [0, 1]])


class U1(_Gate):
    symbol = "U1"

    def __init__(self, *qubits, angle=0, **kwargs):
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self.angle)]])


class Cz(_Gate):
    symbol = "CZ"

    def __init__(self, *qubits, **kwargs):
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


class Rx(_Gate):
    symbol = "RX"

    def __init__(self, *qubits, angle=math.pi / 2, **kwargs):
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [math.cos(self.angle / 2), -math.sin(self.angle / 2) * 1j],
                [-math.sin(self.angle / 2) * 1j, math.cos(self.angle / 2)],
            ]
        )


class Ry(_Gate):
    symbol = "RY"

    def __init__(self, *qubits, angle=math.pi / 2, **kwargs):
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [math.cos(self.angle / 2), -math.sin(self.angle / 2)],
                [math.sin(self.angle / 2), math.cos(self.angle / 2)],
            ]
        )


class Rz(_Gate):
    symbol = "RZ"

    def __init__(self, *qubits, angle, **kwargs):
        self.angle = angle
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[np.exp(-(1 / 2) * 1j * self.angle), 0], [0, np.exp((1 / 2) * 1j * self.angle)]], dtype="complex"
        )


class U3(_Gate):
    symbol = "U3"

    def __init__(self, *qubits, theta=0, phi=0, lam=0, **kwargs):
        kwargs["dimension"] = 1
        self.theta = theta
        self.phi = phi
        self.lam = lam

        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [math.cos(self.theta / 2), -np.exp(1j * self.lam) * math.sin(self.theta / 2)],
                [
                    np.exp(1j * self.phi) * math.sin(self.theta / 2),
                    np.exp(1j * (self.phi + self.lam)) * math.cos(self.theta / 2),
                ],
            ]
        )


class U2(U3):
    symbol = "U2"

    def __init__(self, *qubits, phi=0, lam=0, **kwargs):
        super().__init__(*qubits, theta=np.pi / 2, phi=phi, lam=lam, **kwargs)
        self.symbol = "u2"


class Init_x(_Gate):
    def __init__(self, *qubits, **kwargs):
        self.H = Hadamard(0)
        kwargs["dimension"] = 1
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return self.H.to_matrix()


class Init_y(_Gate):
    def __init__(self, *qubits, **kwargs):
        self.H = Hadamard(0)
        self.S = S()
        kwargs["dimension"] = 1
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return self.S.to_matrix().dot(self.H.to_matrix())


class Cr(_Gate):
    symbol = "CU1"

    def __init__(self, *qubits, angle, **kwargs):
        self.angle = angle
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self.angle)]], dtype="complex")


class CRk(_Gate):
    def __init__(self, *qubits, k, **kwargs):
        self.k = k
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(2 * 1j * np.pi / 2 ** self.k)]], dtype="complex"
        )


# Aliases
H = h = Hadamard
X = x = PauliX
Y = y = PauliY
Z = z = PauliZ
swap = SWAP
Fredkin = cswap = CSWAP
CX = cx = CNOT
