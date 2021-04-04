from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from shor.layers import _Layer
from shor.operations import _Operation
from shor.quantum import QC


def to_qiskit_registers(shor_circuit):
    qbit_registers, cbit_registers = shor_circuit.to_registers()
    args = []
    qiskit_qbit_registers = []
    qiskit_classical_registers = []
    for i in range(0, len(qbit_registers._qbits)):
        tmp_reg = QuantumRegister(len(qbit_registers._qbits[i]), name=qbit_registers._qbits[i][0][1].name)  # + str(i)
        qiskit_qbit_registers.append(tmp_reg)
        args.append(tmp_reg)
    for i in range(0, len(cbit_registers._cbits)):
        tmp_reg = ClassicalRegister(
            len(cbit_registers._cbits[i]), name=cbit_registers._cbits[i][0][1].name
        )  # TODO: Classical bits need to be broken up more I think
        qiskit_classical_registers.append(tmp_reg)
        args.append(tmp_reg)

    return args, qiskit_qbit_registers, qiskit_classical_registers


def to_qiskit_circuit(shor_circuit: QC) -> QuantumCircuit:
    args, qiskit_qbit_registers, qiskit_classical_registers = to_qiskit_registers(shor_circuit)
    qiskit_circuit = QuantumCircuit(*args)

    for gate_or_op in shor_circuit.to_gates(include_operations=True):
        transpile_gate(qiskit_circuit, qiskit_qbit_registers, qiskit_classical_registers, gate_or_op)

    return qiskit_circuit


def to_qiskit_symbol(shor_layer: _Layer) -> str:
    return shor_layer.symbol.lower()


def gate_logic(shor_layer, symbol):
    args = []
    kwargs = {}

    if symbol in ["crx", "cry", "crz", "cu1", "rx", "ry", "rz", "u1"]:
        args.append(shor_layer.angle)
    if symbol in ["u", "u3"]:
        args.append(shor_layer.theta)
    if symbol in ["u", "u2", "u3"]:
        args.append(shor_layer.phi)
        args.append(shor_layer.lam)

    args.extend(shor_layer.qbits)
    return args, kwargs


def transpile_gate(
    qiskit_circuit: QuantumCircuit,
    qiskit_qbit_registers: QuantumRegister,
    qiskit_classical_registers: ClassicalRegister,
    shor_layer: _Layer,
):
    symbol = to_qiskit_symbol(shor_layer)

    if isinstance(shor_layer, _Operation):
        if symbol == "measure":
            # if len(shor_layer.qbits) == len(shor_layer.bits) == qiskit_circuit.size():
            #     qiskit_circuit.measure_active()
            # else:
            for qbit, bit in zip(shor_layer.qbits, shor_layer.bits):
                qiskit_circuit.measure(qbit, bit)
        elif symbol == "barrier":
            qiskit_circuit.barrier()
        elif symbol == "conditional":
            symbol = to_qiskit_symbol(shor_layer.gate)
            for c in filter(
                lambda l: l.name == shor_layer.cbit, qiskit_classical_registers
            ):  # This line currently broken fix and c_if should be close to working
                testing_cbit = c
            args, kwargs = gate_logic(shor_layer, symbol)
            qiskit_circuit.__getattribute__(symbol)(*args, **kwargs).c_if(testing_cbit, shor_layer.logic)

        return

    args, kwargs = gate_logic(shor_layer, symbol)

    qiskit_circuit.__getattribute__(symbol)(*args, **kwargs)
