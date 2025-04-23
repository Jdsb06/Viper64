from dataclasses import dataclass
from enum import Enum
import numpy as np

class InstructionType(Enum):
    R = 0  # Register
    I = 1  # Immediate
    S = 2  # Store
    B = 3  # Branch
    U = 4  # Upper immediate
    J = 5  # Jump

@dataclass
class DecodedInstruction:
    instruction_type: InstructionType
    opcode: int
    rd: int
    funct3: int
    rs1: int
    rs2: int
    funct7: int
    immediate: int
    raw_instruction: int

class InstructionDecoder:
    def __init__(self):
        pass
    
    def decode(self, instruction: int) -> DecodedInstruction:
        """Decode a 32-bit RISC-V instruction"""
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 0x1F
        funct3 = (instruction >> 12) & 0x7
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        funct7 = (instruction >> 25) & 0x7F
        
        # Determine instruction type and extract immediate
        instruction_type = self._get_instruction_type(opcode)
        immediate = self._get_immediate(instruction, instruction_type)
        
        return DecodedInstruction(
            instruction_type=instruction_type,
            opcode=opcode,
            rd=rd,
            funct3=funct3,
            rs1=rs1,
            rs2=rs2,
            funct7=funct7,
            immediate=immediate,
            raw_instruction=instruction
        )
    
    def _get_instruction_type(self, opcode: int) -> InstructionType:
        """Determine instruction type from opcode"""
        if opcode == 0x33:  # R-type (OP, OP-32, M-extension)
            return InstructionType.R
        elif opcode in [0x03, 0x13, 0x67, 0x73]:  # I-type (LOAD, OP-IMM, JALR, SYSTEM)
            return InstructionType.I
        elif opcode == 0x23:  # S-type (STORE)
            return InstructionType.S
        elif opcode == 0x63:  # B-type (BRANCH)
            return InstructionType.B
        elif opcode in [0x37, 0x17]:  # U-type (LUI, AUIPC)
            return InstructionType.U
        elif opcode == 0x6F:  # J-type (JAL)
            return InstructionType.J
        elif opcode in [0x07, 0x27]:  # D-extension (FLD, FSD)
            return InstructionType.I  # Use I-type format for floating-point loads/stores
        elif opcode in [0x43, 0x47, 0x4B, 0x4F]:  # D-extension (FMADD.D, FMSUB.D, FNMSUB.D, FNMADD.D)
            return InstructionType.R  # Use R-type format for floating-point arithmetic
        elif opcode == 0x53:  # D-extension (FOP.D)
            return InstructionType.R  # Use R-type format for floating-point operations
        elif opcode == 0:  # NOP or invalid instruction
            return InstructionType.R  # Treat as R-type to avoid raising exception
        else:
            raise ValueError(f"Unknown opcode: {opcode}")
    
    def _get_immediate(self, instruction: int, instruction_type: InstructionType) -> int:
        """Extract immediate value based on instruction type"""
        if instruction_type == InstructionType.I:
            # I-type: 12-bit signed immediate
            imm = ((instruction >> 20) & 0xFFF)
            # Sign extend if bit 11 is set
            if imm & 0x800:
                imm |= 0xFFFFFFFFFFFFF000
            return np.int64(imm).item()  # Convert to Python int
        elif instruction_type == InstructionType.S:
            # S-type: 12-bit signed immediate
            imm = (((instruction >> 25) & 0x7F) << 5) | ((instruction >> 7) & 0x1F)
            # Sign extend if bit 11 is set
            if imm & 0x800:
                imm |= 0xFFFFFFFFFFFFF000
            return np.int64(imm).item()  # Convert to Python int
        elif instruction_type == InstructionType.B:
            # B-type: 13-bit signed immediate
            imm = (((instruction >> 31) & 0x1) << 12) | \
                  (((instruction >> 7) & 0x1) << 11) | \
                  (((instruction >> 25) & 0x3F) << 5) | \
                  (((instruction >> 8) & 0xF) << 1)
            # Sign extend if bit 12 is set
            if imm & 0x1000:
                imm |= 0xFFFFFFFFFFFFE000
            return np.int64(imm).item()  # Convert to Python int
        elif instruction_type == InstructionType.U:
            # U-type: 20-bit upper immediate (shifted left by 12)
            imm = instruction & 0xFFFFF000
            return np.int64(imm).item()  # Convert to Python int
        elif instruction_type == InstructionType.J:
            # J-type: 21-bit signed immediate
            imm = (((instruction >> 31) & 0x1) << 20) | \
                  (((instruction >> 12) & 0xFF) << 12) | \
                  (((instruction >> 20) & 0x1) << 11) | \
                  (((instruction >> 21) & 0x3FF) << 1)
            # Sign extend if bit 20 is set
            if imm & 0x100000:
                imm |= 0xFFFFFFFFFFE00000
            return np.int64(imm).item()  # Convert to Python int
        else:
            return 0  # R-type has no immediate 