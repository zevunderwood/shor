from collections import Iterable


class _Layer(object):
    """Abstract base quantum layer class"""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Layer")
        pass

    def to_gates(self):
        pass


class Qbits(_Layer, Iterable):
    def __init__(self, num, state=0, **kwargs):
        self.num = num
        self.state = state
        self._qbits = list(range(num))

        super().__init__(name="Qbits ({})".format(str(num)), **kwargs)

    def to_gates(self):
        return []

    def __iter__(self):
        return self._qbits.__iter__()

    def __getitem__(self, key):
        return self._qbits[key]


# Aliases
Qubits = Qbits
