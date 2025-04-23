from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .registers import RegisterFile, FPURegisterFile
from .memory import Memory
from .alu import ALU, ALUOp
from .fpu import FPU, FPUOp
from .control import ControlUnit, ControlSignals
from ..utils.instruction_decoder import InstructionDecoder, DecodedInstruction
from ..utils.hazard_detector import HazardDetector, PipelineRegisters

@dataclass
class PipelineState:
    pc: int
    instruction: int
    control_signals: ControlSignals
    rs1_value: int
    rs2_value: int
    alu_result: int
    memory_data: int
    write_data: int
    stall: bool
    flush: bool
    
    def get_signed_alu_result(self) -> int:
        """Convert ALU result to signed value for display"""
        # Convert from unsigned to signed 64-bit integer
        # This is a direct conversion from uint64 to int64
        return np.int64(self.alu_result)

class Pipeline:
    def __init__(self, memory_size: int = 1024 * 1024):
        self.register_file = RegisterFile()
        self.fpu_register_file = FPURegisterFile()
        self.memory = Memory(memory_size)
        self.alu = ALU()
        self.fpu = FPU()
        self.control_unit = ControlUnit()
        self.instruction_decoder = InstructionDecoder()
        self.hazard_detector = HazardDetector()
        
        self.pc = 0
        self.pipeline_registers = PipelineRegisters()
        self.stall = False
        self.flush = False
        self.data_memory_base = 0x1000  # Base address for a data memory section
        
    def load_program(self, program: bytes, start_address: int = 0) -> None:
        """Load a program into memory"""
        self.memory.load_program(program, start_address)
        self.pc = start_address
        
    def step(self) -> Optional[PipelineState]:
        """Execute one pipeline step"""
        if self.stall:
            return self._handle_stall()
            
        # Initialize variables
        write_data = 0
        instruction = 0
            
        # Instruction Fetch
        if not self.flush:
            # Convert PC to unsigned address
            pc_address = self.pc & 0xFFFFFFFF
            instruction = self.memory.read_word(pc_address)
            self.pipeline_registers.if_id_instruction = instruction
            self.pipeline_registers.if_id_pc = self.pc
            
        # Instruction Decode
        decoded = self.instruction_decoder.decode(self.pipeline_registers.if_id_instruction)
        control_signals = self.control_unit.generate_control_signals(
            decoded.opcode, decoded.funct3, decoded.funct7
        )
        
        # Read register values
        rs1_value = self.register_file.read(decoded.rs1)
        rs2_value = self.register_file.read(decoded.rs2)
        
        # Update ID/EX registers
        self.pipeline_registers.id_ex_reg_write = control_signals.reg_write
        self.pipeline_registers.id_ex_mem_to_reg = control_signals.mem_to_reg
        self.pipeline_registers.id_ex_mem_write = control_signals.mem_write
        self.pipeline_registers.id_ex_branch = control_signals.branch
        self.pipeline_registers.id_ex_alu_op = control_signals.alu_op
        self.pipeline_registers.id_ex_alu_src = control_signals.alu_src
        self.pipeline_registers.id_ex_jump = control_signals.jump
        self.pipeline_registers.id_ex_rs1 = decoded.rs1
        self.pipeline_registers.id_ex_rs2 = decoded.rs2
        self.pipeline_registers.id_ex_rd = decoded.rd
        self.pipeline_registers.id_ex_immediate = decoded.immediate
        self.pipeline_registers.id_ex_pc = self.pipeline_registers.if_id_pc
        self.pipeline_registers.id_ex_instruction = self.pipeline_registers.if_id_instruction
        
        # Update Hazard Detector registers with current pipeline registers
        self.hazard_detector.update_registers(self.pipeline_registers)
        
        # Execute
        # Handle forwarding
        forward_a, forward_b = self.hazard_detector.get_forwarding_paths()
        
        # Select forwarded values if needed
        if forward_a == 1:
            rs1_value = self.pipeline_registers.ex_mem_alu_result
        elif forward_a == 2:
            rs1_value = self.pipeline_registers.mem_wb_alu_result
            
        if forward_b == 1:
            rs2_value = self.pipeline_registers.ex_mem_alu_result
        elif forward_b == 2:
            rs2_value = self.pipeline_registers.mem_wb_alu_result
            
        # ALU operation
        alu_operand2 = (self.pipeline_registers.id_ex_immediate 
                       if self.pipeline_registers.id_ex_alu_src 
                       else rs2_value)
        
        # Get ALU control using fields from ID/EX registers
        alu_control = self.control_unit.get_alu_control(
            self.pipeline_registers.id_ex_alu_op,
            (self.pipeline_registers.id_ex_instruction >> 12) & 0x7,  # funct3 from ID/EX
            (self.pipeline_registers.id_ex_instruction >> 25) & 0x7F  # funct7 from ID/EX
        )
        
        # Handle JALR in execute stage
        if self.pipeline_registers.id_ex_alu_src and self.pipeline_registers.id_ex_jump:
            # Calculate JALR target address
            jump_target = (rs1_value + self.pipeline_registers.id_ex_immediate) & 0xFFFFFFFF
            jump_target = jump_target & ~1  # Clear LSB as per RISC-V spec
            
            # Store return address (PC + 4) in EX/MEM for writeback
            self.pipeline_registers.ex_mem_alu_result = self.pipeline_registers.id_ex_pc + 4
            
            # Update PC for next cycle
            self.pc = jump_target
            
            # Set ALU result for potential forwarding
            alu_result = self.pipeline_registers.ex_mem_alu_result
            
            # Flush pipeline
            self.flush = True
            # Clear pipeline registers
            self.pipeline_registers.if_id_instruction = 0
            self.pipeline_registers.if_id_pc = 0
            # Also clear the ID / EX stage to prevent executing wrong instructions
            self.pipeline_registers.id_ex_reg_write = False
            self.pipeline_registers.id_ex_mem_to_reg = False
            self.pipeline_registers.id_ex_mem_write = False
            self.pipeline_registers.id_ex_branch = False
            self.pipeline_registers.id_ex_alu_op = 0
            self.pipeline_registers.id_ex_alu_src = False
            self.pipeline_registers.id_ex_jump = False
            self.pipeline_registers.id_ex_rs1 = 0
            self.pipeline_registers.id_ex_rs2 = 0
            self.pipeline_registers.id_ex_rd = 0
            self.pipeline_registers.id_ex_immediate = 0
            self.pipeline_registers.id_ex_pc = 0
            self.pipeline_registers.id_ex_instruction = 0
        else:
            # Regular ALU operation
            # Check if instruction is LUI (opcode 0x37)
            if (self.pipeline_registers.id_ex_instruction & 0x7F) == 0x37:
                alu_result = self.pipeline_registers.id_ex_immediate
            else:
                alu_result = self.alu.execute(ALUOp(alu_control), rs1_value, alu_operand2)
        
        # Update EX/MEM registers
        self.pipeline_registers.ex_mem_reg_write = self.pipeline_registers.id_ex_reg_write
        self.pipeline_registers.ex_mem_mem_to_reg = self.pipeline_registers.id_ex_mem_to_reg
        self.pipeline_registers.ex_mem_mem_write = self.pipeline_registers.id_ex_mem_write
        self.pipeline_registers.ex_mem_alu_result = alu_result
        self.pipeline_registers.ex_mem_write_data = rs2_value
        self.pipeline_registers.ex_mem_rd = self.pipeline_registers.id_ex_rd
        
        # Memory Access
        if self.pipeline_registers.ex_mem_mem_write:
            # Convert address to unsigned and add data memory base
            address = (self.pipeline_registers.ex_mem_alu_result & 0xFFFFFFFF) + self.data_memory_base
            
            # Check if this is a floating-point store
            if control_signals.fpu_op is not None and control_signals.fpu_op == 1:  # FSD
                # Convert integer to floating-point for storage
                fp_value = np.float64(np.int64(self.pipeline_registers.ex_mem_write_data))
                # Store as 64-bit value
                self.memory.write_double(address, fp_value)
            else:
                # Regular integer store
                self.memory.write_word(
                    address,
                    self.pipeline_registers.ex_mem_write_data
                )
            
        # Only read from memory if we need to (mem_to_reg is true)
        memory_data = 0  # Default value
        if self.pipeline_registers.ex_mem_mem_to_reg:
            # Convert address to unsigned and add data memory base
            address = (self.pipeline_registers.ex_mem_alu_result & 0xFFFFFFFF) + self.data_memory_base
            
            # Check if this is a floating-point load
            if control_signals.fpu_op is not None and control_signals.fpu_op == 0:  # FLD
                # Load as a 64-bit floating-point value
                fp_value = self.memory.read_double(address)
                # Convert to integer for a register file
                memory_data = np.uint64(np.int64(fp_value))
            else:
                # Regular integer load
                memory_data = self.memory.read_word(address)
            
        # Update MEM/WB registers
        self.pipeline_registers.mem_wb_reg_write = self.pipeline_registers.ex_mem_reg_write
        self.pipeline_registers.mem_wb_mem_to_reg = self.pipeline_registers.ex_mem_mem_to_reg
        self.pipeline_registers.mem_wb_read_data = memory_data
        self.pipeline_registers.mem_wb_alu_result = self.pipeline_registers.ex_mem_alu_result
        self.pipeline_registers.mem_wb_rd = self.pipeline_registers.ex_mem_rd
        
        # Also update FPU-related registers
        self.pipeline_registers.mem_wb_fpu_reg_write = self.pipeline_registers.ex_mem_fpu_reg_write
        self.pipeline_registers.mem_wb_fpu_mem_to_reg = self.pipeline_registers.ex_mem_fpu_mem_to_reg
        self.pipeline_registers.mem_wb_fpu_result = self.pipeline_registers.ex_mem_fpu_result
        self.pipeline_registers.mem_wb_fpu_rd = self.pipeline_registers.ex_mem_fpu_rd
        
        # Special handling for division by zero and multiplication
        if (self.pipeline_registers.id_ex_instruction & 0xFE00707F) == 0x02004033:  # DIV
            if self.pipeline_registers.ex_mem_alu_result == 0xFFFFFFFFFFFFFFFF:
                self.pipeline_registers.mem_wb_alu_result = np.uint64(0xFFFFFFFFFFFFFFFF)
        elif (self.pipeline_registers.id_ex_instruction & 0xFE00707F) == 0x02005033:  # DIVU
            if self.pipeline_registers.ex_mem_alu_result == 0xFFFFFFFFFFFFFFFF:
                self.pipeline_registers.mem_wb_alu_result = np.uint64(0xFFFFFFFFFFFFFFFF)
        # Special handling for multiplication
        elif (self.pipeline_registers.id_ex_instruction & 0xFE00707F) == 0x02000033:  # MUL
            # Ensure the result is properly sign-extended
            self.pipeline_registers.mem_wb_alu_result = np.uint64(np.int64(self.pipeline_registers.ex_mem_alu_result))
        
        # Execute writeback stage
        self.writeback_stage()
            
        # PC Update Logic
        next_pc = self.pc + 4  # Default: increment PC
        
        if control_signals.jump and not control_signals.alu_src:  # JAL only (not JALR)
            # For JAL, store return address (PC + 4) in destination register
            if self.pipeline_registers.id_ex_reg_write:
                self.pipeline_registers.ex_mem_alu_result = self.pc + 4
            
            # Calculate jump target for JAL
            jump_target = (self.pc + decoded.immediate) & 0xFFFFFFFF
            next_pc = jump_target
            self.flush = True
            # Clear pipeline registers for next cycle
            self.pipeline_registers.if_id_instruction = 0
            self.pipeline_registers.if_id_pc = 0
            # Also clear ID/EX stage to prevent executing wrong instructions
            self.pipeline_registers.id_ex_reg_write = False
            self.pipeline_registers.id_ex_mem_to_reg = False
            self.pipeline_registers.id_ex_mem_write = False
            self.pipeline_registers.id_ex_branch = False
            self.pipeline_registers.id_ex_alu_op = 0
            self.pipeline_registers.id_ex_alu_src = False
            self.pipeline_registers.id_ex_rs1 = 0
            self.pipeline_registers.id_ex_rs2 = 0
            self.pipeline_registers.id_ex_rd = 0
            self.pipeline_registers.id_ex_immediate = 0
            self.pipeline_registers.id_ex_pc = 0
            self.pipeline_registers.id_ex_instruction = 0
        elif control_signals.branch:
            # Calculate branch target
            # Sign-extend the immediate value for branch instructions
            immediate = decoded.immediate
            if immediate & 0x800:  # Check if bit 11 is set (negative)
                immediate |= 0xFFFFFFFFFFFFF000  # Sign extend
            branch_target = (self.pc + immediate) & 0xFFFFFFFF  # Ensure target is unsigned
            
            # Validate branch target
            if branch_target >= self.memory.size - 4:
                # If branch target is out of bounds, set it to a safe value
                branch_target = self.memory.size - 4
                print(f"Warning: Branch target {branch_target} out of bounds, redirecting to {self.memory.size - 4}")
            
            # Check branch condition based on ALU result and funct3
            branch_taken = False
            if decoded.funct3 == 0:  # BEQ
                branch_taken = self.alu.get_zero()
            elif decoded.funct3 == 1:  # BNE
                branch_taken = not self.alu.get_zero()
            elif decoded.funct3 == 4:  # BLT
                branch_taken = (self.alu.result & (1 << 63)) != 0  # Check sign bit
            elif decoded.funct3 == 5:  # BGE
                branch_taken = (self.alu.result & (1 << 63)) == 0  # Check sign bit
            elif decoded.funct3 == 6:  # BLTU
                # For BLTU, we need to check if rs1 < rs2 (unsigned)
                # We need to directly compare the unsigned values
                branch_taken = rs1_value < rs2_value
            elif decoded.funct3 == 7:  # BGEU
                # For BGEU, we need to check if rs1 >= rs2 (unsigned)
                # We need to directly compare the unsigned values
                branch_taken = rs1_value >= rs2_value
                
            if branch_taken:
                next_pc = branch_target
                self.flush = True
                # Clear pipeline registers for the next cycle
                self.pipeline_registers.if_id_instruction = 0
                self.pipeline_registers.if_id_pc = 0
                # Also clear the ID / EX stage to prevent executing wrong instructions
                self.pipeline_registers.id_ex_reg_write = False
                self.pipeline_registers.id_ex_mem_to_reg = False
                self.pipeline_registers.id_ex_mem_write = False
                self.pipeline_registers.id_ex_branch = False
                self.pipeline_registers.id_ex_alu_op = 0
                self.pipeline_registers.id_ex_alu_src = False
                self.pipeline_registers.id_ex_rs1 = 0
                self.pipeline_registers.id_ex_rs2 = 0
                self.pipeline_registers.id_ex_rd = 0
                self.pipeline_registers.id_ex_immediate = 0
                self.pipeline_registers.id_ex_pc = 0
                self.pipeline_registers.id_ex_instruction = 0
            else:
                self.flush = False
        else:
            self.flush = False
            
        # Update PC for next cycle
        self.pc = next_pc & 0xFFFFFFFF  # Ensure PC is unsigned
            
        # Return pipeline state
        return PipelineState(
            pc=self.pc,
            instruction=instruction,
            control_signals=control_signals,
            rs1_value=rs1_value,
            rs2_value=rs2_value,
            alu_result=alu_result,
            memory_data=memory_data,
            write_data=write_data,
            stall=self.stall,
            flush=self.flush
        )
    
    def _handle_stall(self) -> Optional[PipelineState]:
        """Handle pipeline stall"""
        self.stall = False
        return None
    
    def reset(self) -> None:
        """Reset the pipeline state"""
        self.register_file.reset()
        self.fpu_register_file.reset()
        self.memory.clear()
        self.pc = 0
        self.pipeline_registers = PipelineRegisters()
        self.stall = False
        self.flush = False

    def writeback_stage(self):
        """Execute the writeback stage of the pipeline."""
        # Get control signals from MEM/WB pipeline register
        reg_write = self.pipeline_registers.mem_wb_reg_write
        fpu_reg_write = self.pipeline_registers.mem_wb_fpu_reg_write
        mem_to_reg = self.pipeline_registers.mem_wb_mem_to_reg
        fpu_mem_to_reg = self.pipeline_registers.mem_wb_fpu_mem_to_reg
        rd = self.pipeline_registers.mem_wb_rd
        fpu_rd = self.pipeline_registers.mem_wb_fpu_rd
        
        # Write back to integer register file
        if reg_write:
            write_data = (self.pipeline_registers.mem_wb_read_data 
                         if mem_to_reg 
                         else self.pipeline_registers.mem_wb_alu_result)
            self.register_file.write(rd, write_data)
            
        # Write back to an FPU register file
        if fpu_reg_write:
            write_data = (self.pipeline_registers.mem_wb_read_data 
                         if fpu_mem_to_reg 
                         else self.pipeline_registers.mem_wb_fpu_result)
            self.fpu_register_file.write(fpu_rd, np.float64(write_data))

class PipelineRegisters:
    """Class to hold pipeline registers."""
    
    def __init__(self):
        # IF/ID Pipeline Registers
        self.if_id_pc = 0
        self.if_id_instruction = 0
        
        # ID/EX Pipeline Registers
        self.id_ex_reg_write = False
        self.id_ex_mem_to_reg = False
        self.id_ex_mem_write = False
        self.id_ex_branch = False
        self.id_ex_alu_src = False
        self.id_ex_alu_op = 0
        self.id_ex_rs1 = 0
        self.id_ex_rs2 = 0
        self.id_ex_rd = 0
        self.id_ex_immediate = 0
        self.id_ex_pc = 0
        self.id_ex_instruction = 0
        
        # FPU Control Signals
        self.id_ex_fpu_reg_write = False
        self.id_ex_fpu_mem_to_reg = False
        self.id_ex_fpu_op = 0
        self.id_ex_fpu_rs1 = 0
        self.id_ex_fpu_rs2 = 0
        self.id_ex_fpu_rd = 0
        self.id_ex_fpu_rs1_data = 0.0
        self.id_ex_fpu_rs2_data = 0.0
        
        # EX/MEM Pipeline Registers
        self.ex_mem_reg_write = False
        self.ex_mem_mem_to_reg = False
        self.ex_mem_mem_write = False
        self.ex_mem_branch = False
        self.ex_mem_alu_result = 0
        self.ex_mem_zero = False
        self.ex_mem_rd = 0
        self.ex_mem_write_data = 0
        
        # FPU Results
        self.ex_mem_fpu_reg_write = False
        self.ex_mem_fpu_mem_to_reg = False
        self.ex_mem_fpu_result = 0.0
        self.ex_mem_fpu_rd = 0
        
        # MEM/WB Pipeline Registers
        self.mem_wb_reg_write = False
        self.mem_wb_mem_to_reg = False
        self.mem_wb_read_data = 0
        self.mem_wb_alu_result = 0
        self.mem_wb_rd = 0
        
        # FPU Write-back
        self.mem_wb_fpu_reg_write = False
        self.mem_wb_fpu_mem_to_reg = False
        self.mem_wb_fpu_result = 0.0
        self.mem_wb_fpu_rd = 0 

