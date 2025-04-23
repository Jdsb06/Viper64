from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Opcode(Enum):
    # RV64I Base Integer Instructions
    LOAD = 0x03
    STORE = 0x23
    BRANCH = 0x63
    JALR = 0x67
    JAL = 0x6F
    OP_IMM = 0x13
    OP = 0x33
    SYSTEM = 0x73
    AUIPC = 0x17
    LUI = 0x37
    
    # RV64M Extension
    MUL = 0x33  # Same as OP, distinguished by funct3/funct7
    DIV = 0x33
    DIVU = 0x33
    REM = 0x33
    REMU = 0x33
    
    # RV64D Extension
    FLD = 0x07
    FSD = 0x27
    FMADD_D = 0x43
    FMSUB_D = 0x47
    FNMSUB_D = 0x4B
    FNMADD_D = 0x4F
    FOP_D = 0x53

@dataclass
class ControlSignals:
    reg_write: bool = False
    mem_to_reg: bool = False
    mem_write: bool = False
    branch: bool = False
    alu_op: int = 0
    alu_src: bool = False
    jump: bool = False
    fpu_op: Optional[int] = None
    fpu_reg_write: bool = False
    fpu_mem_to_reg: bool = False

class ControlUnit:
    def __init__(self):
        self.signals = ControlSignals()
    
    def generate_control_signals(self, opcode: int, funct3: int, funct7: int) -> ControlSignals:
        """Generate control signals based on instruction fields"""
        self.signals = ControlSignals()
        
        # Default values
        self.signals.reg_write = False
        self.signals.mem_to_reg = False
        self.signals.mem_write = False
        self.signals.branch = False
        self.signals.alu_op = 0
        self.signals.alu_src = False
        self.signals.jump = False
        self.signals.fpu_op = None
        self.signals.fpu_reg_write = False
        self.signals.fpu_mem_to_reg = False
        
        # R-type instructions
        if opcode == Opcode.OP.value:
            self.signals.reg_write = True
            if funct7 == 0x01:  # M-extension
                if funct3 == 0: self.signals.alu_op = 12  # MUL
                elif funct3 == 1: self.signals.alu_op = 13  # MULH
                elif funct3 == 2: self.signals.alu_op = 14  # MULHSU
                elif funct3 == 3: self.signals.alu_op = 15  # MULHU
                elif funct3 == 4: self.signals.alu_op = 16  # DIV
                elif funct3 == 5: self.signals.alu_op = 17  # DIVU
                elif funct3 == 6: self.signals.alu_op = 18  # REM
                elif funct3 == 7: self.signals.alu_op = 19  # REMU
            else:  # Regular R-type
                if funct3 == 0:  # ADD/SUB
                    if funct7 == 0x00: self.signals.alu_op = 0  # ADD
                    elif funct7 == 0x20: self.signals.alu_op = 1  # SUB
                elif funct3 == 1: self.signals.alu_op = 4  # SLL
                elif funct3 == 2: self.signals.alu_op = 5  # SLT
                elif funct3 == 3: self.signals.alu_op = 6  # SLTU
                elif funct3 == 4: self.signals.alu_op = 7  # XOR
                elif funct3 == 5:  # SRL/SRA
                    if funct7 == 0x00: self.signals.alu_op = 8  # SRL
                    elif funct7 == 0x20: self.signals.alu_op = 9  # SRA
                elif funct3 == 6: self.signals.alu_op = 10  # OR
                elif funct3 == 7: self.signals.alu_op = 11  # AND
        
        # I-type instructions
        elif opcode == Opcode.OP_IMM.value:
            self.signals.reg_write = True
            self.signals.alu_src = True
            if funct3 == 0:  # ADDI
                self.signals.alu_op = 0  # Add operation
            elif funct3 == 1:  # SLLI
                self.signals.alu_op = 4  # Shift left operation
            elif funct3 == 2:  # SLTI
                self.signals.alu_op = 5  # Set less than signed operation
            elif funct3 == 3:  # SLTIU
                self.signals.alu_op = 6  # Set less than an unsigned operation
            elif funct3 == 4:  # XORI
                self.signals.alu_op = 7  # XOR operation
            elif funct3 == 5:  # SRLI/SRAI
                if funct7 == 0x00:  # SRLI
                    self.signals.alu_op = 8  # Shift right logical operation
                else:  # SRAI
                    self.signals.alu_op = 9  # Shift right arithmetic operation
            elif funct3 == 6:  # ORI
                self.signals.alu_op = 10  # OR operation
            elif funct3 == 7:  # ANDI
                self.signals.alu_op = 11  # AND operation
            else:
                self.signals.alu_op = 0  # Default to ADD for unknown I-type
        
        # Load instructions
        elif opcode == Opcode.LOAD.value:
            self.signals.reg_write = True
            self.signals.mem_to_reg = True
            self.signals.alu_src = True
            self.signals.alu_op = 0  # Add for address calculation
            
        # Store instructions
        elif opcode == Opcode.STORE.value:
            self.signals.mem_write = True
            self.signals.alu_src = True
            self.signals.alu_op = 0  # Add for address calculation
            
        # Branch instructions
        elif opcode == Opcode.BRANCH.value:
            self.signals.branch = True
            self.signals.alu_op = 1  # Branch comparison
            
        # Jump instructions
        elif opcode == Opcode.JAL.value:
            self.signals.reg_write = True
            self.signals.jump = True
            
        elif opcode == Opcode.JALR.value:
            self.signals.reg_write = True
            self.signals.jump = True
            self.signals.alu_src = True
            self.signals.alu_op = 0  # Add for address calculation
            
        # U-type instructions
        elif opcode in [Opcode.LUI.value, Opcode.AUIPC.value]:
            self.signals.reg_write = True
            self.signals.alu_src = True
            if opcode == Opcode.LUI.value:
                # For LUI, we need to pass the immediate value directly
                self.signals.alu_op = 0  # Pass immediate
            else:
                # For AUIPC, we need to add PC to the immediate
                self.signals.alu_op = 0  # Add for address calculation
            
        # D-extension instructions
        elif opcode == Opcode.FLD.value:
            self.signals.reg_write = False
            self.signals.mem_to_reg = True
            self.signals.alu_src = True
            self.signals.alu_op = 0  # Add for address calculation
            self.signals.fpu_op = 0  # Load double
            self.signals.fpu_reg_write = True
            self.signals.fpu_mem_to_reg = True
            
        elif opcode == Opcode.FSD.value:
            self.signals.mem_write = True
            self.signals.alu_src = True
            self.signals.alu_op = 0  # Add for address calculation
            self.signals.fpu_op = 1  # Store double
            self.signals.fpu_reg_write = False
            self.signals.fpu_mem_to_reg = False
            
        elif opcode in [Opcode.FMADD_D.value, Opcode.FMSUB_D.value,
                       Opcode.FNMSUB_D.value, Opcode.FNMADD_D.value]:
            self.signals.reg_write = False
            self.signals.fpu_reg_write = True
            self.signals.fpu_mem_to_reg = False
            self.signals.fpu_op = opcode - Opcode.FMADD_D.value + 2  # 2-5 for FMADD, FMSUB, FNMSUB, FNMADD
            
        elif opcode == Opcode.FOP_D.value:
            self.signals.reg_write = False
            self.signals.fpu_reg_write = True
            self.signals.fpu_mem_to_reg = False
            if funct3 == 0:  # FADD.D
                self.signals.fpu_op = 6
            elif funct3 == 1:  # FSUB.D
                self.signals.fpu_op = 7
            elif funct3 == 2:  # FMUL.D
                self.signals.fpu_op = 8
            elif funct3 == 3:  # FDIV.D
                self.signals.fpu_op = 9
            elif funct3 == 4:  # FMIN.D
                self.signals.fpu_op = 10
            elif funct3 == 5:  # FMAX.D
                self.signals.fpu_op = 11
            elif funct3 == 6:  # FCVT.W.D/FCVT.D.W
                if (funct7 >> 2) & 0x1F == 0:  # FCVT.W.D
                    self.signals.reg_write = True
                    self.signals.fpu_reg_write = False
                    self.signals.fpu_op = 16
                else:  # FCVT.D.W
                    self.signals.reg_write = False
                    self.signals.fpu_reg_write = True
                    self.signals.fpu_op = 18
            elif funct3 == 7:  # FMV.X.D/FMV.D.X/FCLASS.D
                if funct7 == 0x70:  # FMV.X.D
                    self.signals.reg_write = True
                    self.signals.fpu_reg_write = False
                    self.signals.fpu_op = 16  # Treat as FCVT.W.D
                elif funct7 == 0x78:  # FMV.D.X
                    self.signals.reg_write = False
                    self.signals.fpu_reg_write = True
                    self.signals.fpu_op = 18  # Treat as FCVT.D.W
            
        return self.signals
    
    def get_alu_control(self, alu_op: int, funct3: int, funct7: int) -> int:
        """Generate ALU control signal based on ALUOp and instruction fields"""
        if alu_op == 0:  # Load/Store/ADDI/JALR/AUIPC/LUI
            return 0  # Add (or pass through for LUI)
        elif alu_op == 1:  # Branch
            # For branches, ALU always performs subtraction.
            # The pipeline branch logic needs to check funct3 to interpret the result.
            return 1  # Subtract
        elif alu_op >= 4:  # I-type ALU operations (direct mapping)
            return alu_op  # The alu_op value is already the correct operation code
        else:
            return 0  # Default ADD for unknown alu_op