from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PipelineRegisters:
    # IF/ID stage registers
    if_id_pc: int = 0
    if_id_instruction: int = 0
    
    # ID/EX stage registers
    id_ex_reg_write: bool = False
    id_ex_mem_to_reg: bool = False
    id_ex_mem_write: bool = False
    id_ex_branch: bool = False
    id_ex_alu_op: int = 0
    id_ex_alu_src: bool = False
    id_ex_rs1: int = 0
    id_ex_rs2: int = 0
    id_ex_rd: int = 0
    id_ex_immediate: int = 0
    id_ex_pc: int = 0
    id_ex_instruction: int = 0
    
    # EX/MEM stage registers
    ex_mem_reg_write: bool = False
    ex_mem_mem_to_reg: bool = False
    ex_mem_mem_write: bool = False
    ex_mem_alu_result: int = 0
    ex_mem_write_data: int = 0
    ex_mem_rd: int = 0
    
    # MEM/WB stage registers
    mem_wb_reg_write: bool = False
    mem_wb_mem_to_reg: bool = False
    mem_wb_read_data: int = 0
    mem_wb_alu_result: int = 0
    mem_wb_rd: int = 0

class HazardDetector:
    def __init__(self):
        self.registers = PipelineRegisters()
    
    def detect_data_hazard(self) -> Tuple[bool, bool, bool]:
        """
        Detect data hazards and return forwarding control signals
        Returns: (forward_a, forward_b, stall)
        """
        forward_a = 0
        forward_b = 0
        stall = False
        
        # Check for RAW hazards
        if self.registers.id_ex_reg_write and self.registers.id_ex_rd != 0:
            # EX hazard
            if self.registers.id_ex_rd == self.registers.id_ex_rs1:
                forward_a = 1
            if self.registers.id_ex_rd == self.registers.id_ex_rs2:
                forward_b = 1
                
            # MEM hazard
            if self.registers.ex_mem_reg_write and self.registers.ex_mem_rd != 0:
                if self.registers.ex_mem_rd == self.registers.id_ex_rs1:
                    forward_a = 2
                if self.registers.ex_mem_rd == self.registers.id_ex_rs2:
                    forward_b = 2
                    
        # Check for load-use hazard
        if self.registers.id_ex_mem_to_reg and self.registers.id_ex_rd != 0:
            if (self.registers.id_ex_rd == self.registers.id_ex_rs1 or 
                self.registers.id_ex_rd == self.registers.id_ex_rs2):
                stall = True
                
        # Check for branch data hazard
        if self.registers.id_ex_branch:
            if (self.registers.ex_mem_reg_write and 
                self.registers.ex_mem_rd != 0 and
                (self.registers.ex_mem_rd == self.registers.id_ex_rs1 or
                 self.registers.ex_mem_rd == self.registers.id_ex_rs2)):
                stall = True
                
        # Check for WAW hazard
        if (self.registers.id_ex_reg_write and 
            self.registers.ex_mem_reg_write and
            self.registers.id_ex_rd == self.registers.ex_mem_rd):
            stall = True
            
        # Check for WAR hazard
        if (self.registers.id_ex_reg_write and
            self.registers.mem_wb_reg_write and
            self.registers.id_ex_rd == self.registers.mem_wb_rd):
            stall = True
                
        return forward_a, forward_b, stall
    
    def detect_control_hazard(self) -> bool:
        """Detect control hazards (branches and jumps)"""
        return self.registers.id_ex_branch
    
    def update_registers(self, new_registers: PipelineRegisters) -> None:
        """Update pipeline registers"""
        self.registers = new_registers
    
    def get_forwarding_paths(self) -> Tuple[int, int]:
        """Get forwarding paths for ALU operands"""
        forward_a, forward_b, _ = self.detect_data_hazard()
        return forward_a, forward_b
    
    def should_stall(self) -> bool:
        """Determine if pipeline should stall"""
        _, _, stall = self.detect_data_hazard()
        return stall
    
    def should_flush(self) -> bool:
        """Determine if pipeline should flush"""
        return self.detect_control_hazard() 