import math
from typing import Iterable, Union

import numpy as np

from shor.errors import CircuitError
from shor.layers import _Layer
from shor.utils.collections import flatten

QbitOrIterable = Union[int, Iterable]


class _Gate(_Layer):
    """Abstract base quantum gate class

    input_length = valid length of input qubits
    qubits = indices of qubits, to be used as input to gate.

    Attributes
    __________
    input_length : int
       valid length of input qubits
    qubits : int
       indices of qubits, to be used as input to gate

    Methods
    -------
    symbol(self): Returns matrix symbol as lower case for provider transpiler

    qubits(self):Returns qbit indices associated with applying the gate

    to_gates(self):Returns number of gate objects applied to input qubit array. Ex: H([1,2]) will
        return two hadamard gates objects applied to qubits indexed to 1 and 2

    num_states(self):Returns the number of states associated with a state vector and its gate matrix

    to_matrix(self): Return matrix form of gate object

    matrix(self): Call to_matrix which returns matrix form of gate object

    invert(self): Return self

    I(self): Calls invert(self)
    """

    @property
    def symbol(self):
        return self.__class__.__name__.lower()

    def __init__(self, *qbits: QbitOrIterable, **kwargs):
        """
         Parameters
         __________
         qubits : int
             qubits to which the gate is being applied
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        super().__init__(**kwargs)
        self.qbits = flatten(qbits) if qbits else [0]
        self.dimension = kwargs.get("dimension", 1)

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
        """Returns gate objects applied to provided qubit indices
        Needs no inputs

        Returns
        -------
        _Gate Object
            The _Gate Objects applied to the provided qubits. Ex: H([1,2]) will return two gate objects such as
            H(1) and H(2)
        """
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


class CNOT(_Gate):
    """
    Apply the controlled X gate to control and target qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        list of length 2 containing qubit indices to which the controlled X gate is applied
    dimension : int
        number of qubits to which the controlled X gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CX"

    def __init__(self, *qubits, **kwargs):
        """
         Parameters
         __________
         qubits : int
             qubit indices apply CNOT gate. First is control
             qubit and the second applies the Pauli-X gate to the target qubit.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class CY(_Gate):
    """
    Apply the controlled Y gate to control and target qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
     qubits : int
         list of length 2 containing qubit indices controlled Y gate is applied to
    dimension : int
        number of qubits to which the controlled Y gate is applied
    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CY"

    def __init__(self, *qubits, **kwargs):
        """
         Parameters
         __________
         qubits : int
             qubit indices apply CY gate. First is control
             qubit and the second applies the Pauli-Y gate to the target qubit.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1 * 1j], [0, 0, 1j, 0]])


class CSWAP(_Gate):
    """
    Apply the Fredkin gate or sometimes called the CSWAP gate to three qubits
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices CSWAP gate is applied to: the first being a control qubit and the
        SWAP gate is applied to the second and third target qubits.
    dimension : int
        number of qubits to which the CSWAP gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CSWAP"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply CSWAP gate. First is control qubits and the second/third applies the SWAP gate.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 3
        if not qubits:
            qubits = [0, 1, 2]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        cswap_matrix = np.eye(8)
        cswap_matrix[:, [5, 6]] = cswap_matrix[:, [6, 5]]
        return cswap_matrix


class Hadamard(_Gate):
    """
    Apply the hadamard gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the hadamard gate is applied
    dimension : int
        number of qubits to which the hardamard gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "H"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which the hadamard gate is applied.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.multiply(np.divide(1, np.sqrt(self.num_states)), np.array([[1, 1], [1, -1]]))


class PauliX(_Gate):
    """
    Apply the PauliX gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the PauliX gate is applied
    dimension : int
        number of qubits to which the PauliX gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "X"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which the PauliX gate is applied
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[0, 1], [1, 0]])


class PauliY(_Gate):
    """
    Apply the PauliY gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices to which the PauliY gate is applied
    dimension : int
        number of qubits to which the PauliY gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "Y"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which the PailiY gate is applied
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[0, -1j], [1j, 0]])


class PauliZ(_Gate):
    """
    Apply the PauliZ gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices to which the PauliZ gate is applied
    dimension : int
        number of qubits to which the PauliZ gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "Z"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which the PauliZ gate is appliedo
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[1, 0], [0, -1]])


class QFT(_Gate):
    """
    Apply the Quantumn Fourier Transform gate to two qubits
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices to which the QFT gate is applied
    dimension : int
        number of qubits to which the QFT gate is applied

    Methods
    -------
    get_nth_unity_root(self,k)
    to_matrixSelf) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    # TODO: add documentation of class methods
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
    """
    Apply the SWAP gate to two qubits
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices SWAP gate is applied to
    dimension : int
        number of qubits SWAP gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "SWAP"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index SWAP gate is applied to
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2**n where n is the number of qubits
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class Cx(_Gate):
    """
    Apply the the Cx gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices Cx gate is applied to
    dimension : int
        number of qubits Cx gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CX"

    def __init__(self, *qubits, **kwargs):
        """
         Parameters
         __________
         qubits : int
             qubit indexes apply Cx gate. First is the control
             qubit and second the target to which the X gate is applied.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 4x4
        """
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


class CCNOT(_Gate):
    """
    Apply the CCNOT or CCX gate to three qubits. The first and second qubit being control qits and
    the paulix gate is applied to the third qubit.
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices to which CCNOT gate is applied
    dimension : int
        number of qubits CCNOT gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CCX"

    def __init__(self, *qubits, **kwargs):
        """
         Parameters
         __________
         qubits : int
             qubit indices apply CCNOT gate. First and second are control
             qubits and the third applies the Pauli-x gate to the third qubit.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 3
        if not qubits:
            qubits = [0, 1, 2]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 9x9
        """
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
    """
    Apply the CRZ gate to two qubits. The first being the control qubit
    and the Rz gate is applied to the target qubit.
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index CRZ gate is applied to
    angle : float
        angle by which the CRZ gate rotates the second target qubit around the z-axis
    dimension : int
        number of qubits CRZ gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CRZ"

    def __init__(self, *qubits, angle=0, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply CRZ gate. First is a control
            qubit and the second applies the RZ gate to the third qubit.
        angle : float
            angle by which the CRZ gate rotates the second target qubit around the z-axis
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        self.angle = angle
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 4x4
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * self.angle / 2), 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ]
        )


class CH(_Gate):
    """
    Apply the CH gate to two qubits. The first being the control qubit
    and the second applies the hadarmard gate to the target qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index CH gate is applied to
    dimension : int
        number of qubits CH gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CH"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply CH gate. First is control
            qubit second applies the H gate to the target qubit.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 4x4
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )


class S(_Gate):
    """
    Apply the S gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index S gate is applied to
    dimension : int
        number of qubits S gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "S"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index S gate is applied to
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, 1j]])


class Sdg(_Gate):
    """
    Apply the Sdg gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the Sdg gate is applied
    dimension : int
        number of qubits to which the Sdg gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "Sdg"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index Sdg gate is applied to
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, -1j]])


class T(_Gate):
    """
    Apply the T gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the T gate is applied
    dimension : int
        number of qubits to which the T gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "T"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which T gate is applied
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class Tdg(_Gate):
    """
    Apply the Tdg gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which Tdg gate is applied
    dimension : int
        number of qubits to which Tdg gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "Tdg"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index to which Tdg gate is applied
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])


class ID(_Gate):
    """
    Apply the identity (ID) gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which ID gate is applied
    dimension : int
        number of qubits to which ID gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "I"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit index ID gate is applied to
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, 1]])


class U1(_Gate):
    """
    Apply the U1 gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index U1 gate is applied to
    angle : float
        anlge used to rotae qubit of choice around the z-axis
    dimension : int
        number of qubits U1 gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "U1"

    def __init__(self, *qubits, angle=0, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply CRZ gate. First is a control
            qubit and the second applies the RZ gate to the third qubit.
        angle : float
            anlge used to rotae qubit of choice around the z-axis
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0], [0, np.exp(1j * self.angle)]])


class Cz(_Gate):
    """
    Apply the Cz gate to a two qubits. First is control qbit and PauliZ gate is applied to the target qbit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index U1 gate is applied to
    dimension : int
        number of qubits U1 gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CZ"

    def __init__(self, *qubits, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply CZ gate. First is a control
            qubit and the Z gate to the target qubit.
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0, 1]

        super().__init__(*qubits, **kwargs)

    @staticmethod
    def to_matrix() -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 4x4
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


class Rx(_Gate):
    """
    Apply the Rx gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index tow which Rx gate is applied
    dimension : int
        number of qubits to which Rx gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "RX"

    def __init__(self, *qubits, angle=math.pi / 2, **kwargs):
        """
        Parameters
        __________
        qubits : integer
            qubit being rotated
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array(
            [
                [math.cos(self.angle / 2), -math.sin(self.angle / 2) * 1j],
                [-math.sin(self.angle / 2) * 1j, math.cos(self.angle / 2)],
            ]
        )


class Ry(_Gate):
    """
    Apply the Ry gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the Ry gate is applied
    dimension : int
        number of qubits to which the Ry gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "RY"

    def __init__(self, *qubits, angle=math.pi / 2, **kwargs):
        """
        Parameters
        __________
        qubits : integer
            qubit being rotated
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        self.angle = angle
        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array(
            [
                [math.cos(self.angle / 2), -math.sin(self.angle / 2)],
                [math.sin(self.angle / 2), math.cos(self.angle / 2)],
            ]
        )


class Rz(_Gate):
    """
    Apply the Rz gate to a single qubit
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the Rz gate is applied
    dimension : int
        number of qubits to which the Rz gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "RZ"

    def __init__(self, *qubits, angle, **kwargs):
        """
        Parameters
        __________
        qubits : integer
            qubit being rotated
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        self.angle = angle
        kwargs["dimension"] = 1
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array(
            [[np.exp(-(1 / 2) * 1j * self.angle), 0], [0, np.exp((1 / 2) * 1j * self.angle)]], dtype="complex"
        )


class U3(_Gate):
    """
    apply U3 gate for single qubit rotation with 3 euler angles
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the Rz gate is applied
    theta : float
        first angle used to rotae single qubit
    phi : float
        second angle used to rotae single qubit
    lam : float
        third angle used to rotae single qubit
    dimension : int
        number of qubits to which the Rz gate is applied

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "U3"

    def __init__(self, *qubits, theta=0, phi=0, lam=0, **kwargs):
        """
        Parameters
        __________
        qubits : integer
            qubit being rotated
        theta : float
            angle used to rotate single qubit
        phi : float
            angle used to rotate single qubit
        lam : float
            angle used to rotate single qubit
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        kwargs["dimension"] = 1
        self.theta = theta
        self.phi = phi
        self.lam = lam

        if not qubits:
            qubits = [0]

        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
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
    """
    apply U2 gate for single qubit rotation with 2 euler angles
    Inherits from U3 class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index Rz gate is applied to
    phi : float
        second angle used to rotae single qubit
    lam : float
        third angle used to rotae single qubit
    dimension : int
        number of qubits Rz gate is applied to

    Methods
    -------
    Inherits to_matrix froom U3
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "U2"

    def __init__(self, *qubits, phi=0, lam=0, **kwargs):
        """
        Parameters
        __________
        qubits : integer
            qubit being rotated
        phi : float
            angle used to rotate single qubit
        lam : float
            angle used to rotate single qubit
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        super().__init__(*qubits, theta=np.pi / 2, phi=phi, lam=lam, **kwargs)
        self.symbol = "u2"


class Init_x(_Gate):
    """
    Initialize the qubits along x axis or |+> basis
    Inherits from gate class
    Currently not implimented in qiskit and can only be used on shor simulator

    Attributes
    __________
    qubits : int
        qubit index init_x gate is applied to
    dimension : int
        number of qubits init_x gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    def __init__(self, *qubits, **kwargs):
        self.H = Hadamard(0)
        kwargs["dimension"] = 1
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return self.H.to_matrix()


class Init_y(_Gate):
    """
    Initialize the qubits along y axis or |-> basis
    Inherits from gate class
    Currently not implimented in qiskit and can only be used on shor simulator

    Attributes
    __________
    qubits : int
        qubit index init_y gate is applied to
    dimension : int
        number of qubits init_y gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    def __init__(self, *qubits, **kwargs):
        self.H = Hadamard(0)
        self.S = S()
        kwargs["dimension"] = 1
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        return self.S.to_matrix().dot(self.H.to_matrix())


class Cr(_Gate):
    """
    Apply the control phase shift gate Cr or sometimes referred as CU1. First qubit being the control and second target qubit
    has a phase shift applied.
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit index to which the Cr gate is applied
    angle : float
        Floating point number containing angle of phase shift
    dimension : int
        number of qubits Cr gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    symbol = "CU1"

    def __init__(self, *qubits, angle, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indexes apply Cr gate. First is a control
            qubit and second is target qubit.
        angle : float
            anlge used to rotae qubit of choice around the z-axis
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        self.angle = angle
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self.angle)]], dtype="complex")


class CRk(_Gate):
    """
    Apply the parameteruzed control phase shift gate Crk with angle pi/2**k. First qubit being the control and second target qubit
    has a phase shift applied.
    Inherits from gate class

    Attributes
    __________
    symbol : str
        a string used to represent gate for provider transpiler
    qubits : int
        qubit indices to which Crk gate is applied
    angle : float
        Floating point number containing angle of phase shift
    dimension : int
        number of qubits Cr gate is applied to

    Methods
    -------
    to_matrix(self) -> np.ndarray
        Returns matrix form of gate as numpy array
    """

    def __init__(self, *qubits, k, **kwargs):
        """
        Parameters
        __________
        qubits : int
            qubit indices to which Cr gate is applied. First is a control
            qubit and second is target qubit.
        angle : float
            anlge used to rotae qubit of choice around the z-axis
        kwargs : keyword str containing value int
            keyword - dimension containing value int 2**n where n is number of qubits
        """
        self.k = k
        kwargs["dimension"] = 2
        if not qubits:
            qubits = [0]
        super().__init__(*qubits, **kwargs)

    def to_matrix(self) -> np.ndarray:
        """Returns matrix form of gate as numpy array
        Needs no inputs

        Returns
        -------
        numpy array
            matrix form of gate with the size 2x2
        """
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
