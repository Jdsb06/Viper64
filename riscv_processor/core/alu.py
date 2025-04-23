from enum import Enum
import numpy as np

class ALUOp(Enum):
    ADD = 0
    SUB = 1
    SLL = 4
    SLT = 5
    SLTU = 6
    XOR = 7
    SRL = 8
    SRA = 9
    OR = 10
    AND = 11
    MUL = 12
    MULH = 13
    MULHSU = 14
    MULHU = 15
    DIV = 16
    DIVU = 17
    REM = 18
    REMU = 19

class ALU:
    def __init__(self):
        self.result: np.uint64 = np.uint64(0)
        self.zero: bool = False
        
    def execute(self, op: ALUOp, operand1: np.uint64, operand2: np.uint64) -> np.uint64:
        """Execute ALU operation"""
        # Convert operands to signed integers for arithmetic operations
        op1_signed = np.int64(operand1)
        op2_signed = np.int64(operand2)
        
        if op == ALUOp.ADD:
            # Signed addition
            result_signed = op1_signed + op2_signed
            self.result = np.uint64(result_signed)
        elif op == ALUOp.SUB:
            # Signed subtraction
            result_signed = op1_signed - op2_signed
            self.result = np.uint64(result_signed)
        elif op == ALUOp.SLL:
            # Logical shift left (unsigned)
            shift_amount = operand2 & 0x3F  # Ensure shift amount is in valid range
            self.result = operand1 << shift_amount
        elif op == ALUOp.SLT:
            # Set if less than (signed)
            self.result = np.uint64(1 if op1_signed < op2_signed else 0)
        elif op == ALUOp.SLTU:
            # Set if less than (unsigned)
            self.result = np.uint64(1 if operand1 < operand2 else 0)
        elif op == ALUOp.XOR:
            # Bitwise XOR
            self.result = operand1 ^ operand2
        elif op == ALUOp.SRL:
            # Logical shift right (unsigned)
            shift_amount = operand2 & 0x3F  # Ensure shift amount is in valid range
            self.result = operand1 >> shift_amount
        elif op == ALUOp.SRA:
            # Arithmetic shift right (signed)
            shift_amount = operand2 & 0x3F  # Ensure shift amount is in valid range
            self.result = np.uint64(op1_signed >> shift_amount)
        elif op == ALUOp.OR:
            # Bitwise OR
            self.result = operand1 | operand2
        elif op == ALUOp.AND:
            # Bitwise AND
            self.result = operand1 & operand2
        elif op == ALUOp.MUL:
            # Signed multiplication
            result_signed = op1_signed * op2_signed
            self.result = np.uint64(result_signed)
        elif op == ALUOp.MULH:
            # High bits of signed multiplication
            op1 = np.int64(operand1)
            op2 = np.int64(operand2)
            result = np.int128(op1) * np.int128(op2)
            self.result = np.uint64(result >> 64)
        elif op == ALUOp.MULHSU:
            # High bits of signed-unsigned multiplication
            op1 = np.int64(operand1)
            op2 = np.uint64(operand2)
            result = np.int128(op1) * np.uint128(op2)
            self.result = np.uint64(result >> 64)
        elif op == ALUOp.MULHU:
            # High bits of unsigned multiplication
            result = np.uint128(operand1) * np.uint128(operand2)
            self.result = np.uint64(result >> 64)
        elif op == ALUOp.DIV:
            # Signed division
            if op2_signed == 0:
                # Division by zero
                self.result = np.uint64(0xFFFFFFFFFFFFFFFF)  # All bits set to 1
            else:
                result_signed = op1_signed // op2_signed
                self.result = np.uint64(result_signed)
        elif op == ALUOp.DIVU:
            # Unsigned division
            if operand2 == 0:
                # Division by zero
                self.result = np.uint64(0xFFFFFFFFFFFFFFFF)  # All bits set to 1
            else:
                self.result = operand1 // operand2
        elif op == ALUOp.REM:
            # Signed a remainder
            if op2_signed == 0:
                # Division by zero
                self.result = np.uint64(0xFFFFFFFFFFFFFFFF)  # All bits set to 1
            else:
                result_signed = op1_signed % op2_signed
                self.result = np.uint64(result_signed)
        elif op == ALUOp.REMU:
            # Unsigned remainder
            if operand2 == 0:
                # Division by zero
                self.result = np.uint64(0xFFFFFFFFFFFFFFFF)  # All bits set to 1
            else:
                self.result = operand1 % operand2
        else:
            # Default to ADD for unknown operations
            result = np.zeros(1, dtype=np.uint64)
            np.add(operand1, operand2, out=result, casting='unsafe')
            self.result = result[0]
            
        # Set zero flag for branch instructions
        self.zero = (self.result == 0)
        
        return self.result
        
    def get_result(self) -> np.uint64:
        """Get the result of the last ALU operation"""
        return self.result
        
    def get_zero(self) -> bool:
        """Get the zero flag for branch instructions"""
        return self.zero 