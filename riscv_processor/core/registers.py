from typing import List
import numpy as np

class RegisterFile:
    def __init__(self):
        # Initialize 32 64-bit registers (x0-x31)
        self.registers: List[np.uint64] = [np.uint64(0) for _ in range(32)]
        # x0 is hardwired to 0
        self.registers[0] = np.uint64(0)
        
    def read(self, reg_num: int) -> np.uint64:
        """Read from a register"""
        if not 0 <= reg_num < 32:
            raise ValueError(f"Invalid register number: {reg_num}")
        return self.registers[reg_num]
    
    def write(self, reg_num: int, value: np.uint64) -> None:
        """Write to a register"""
        if not 0 <= reg_num < 32:
            raise ValueError(f"Invalid register number: {reg_num}")
        if reg_num == 0:  # x0 is hardwired to 0
            return
        self.registers[reg_num] = value
    
    def get_state(self) -> List[np.uint64]:
        """Get the current state of all registers"""
        return self.registers.copy()
    
    def reset(self) -> None:
        """Reset all registers to 0"""
        self.registers = [np.uint64(0) for _ in range(32)]

class FPURegisterFile:
    def __init__(self):
        # Initialize 32 64-bit floating-point registers (f0-f31)
        self.registers: List[np.float64] = [np.float64(0.0) for _ in range(32)]
        # f0 is hardwired to 0.0
        self.registers[0] = np.float64(0.0)
        
    def read(self, reg_num: int) -> np.float64:
        """Read from a floating-point register"""
        if not 0 <= reg_num < 32:
            raise ValueError(f"Invalid floating-point register number: {reg_num}")
        return self.registers[reg_num]
    
    def write(self, reg_num: int, value: np.float64) -> None:
        """Write to a floating-point register"""
        if not 0 <= reg_num < 32:
            raise ValueError(f"Invalid floating-point register number: {reg_num}")
        if reg_num == 0:  # f0 is hardwired to 0.0
            return
        self.registers[reg_num] = value
    
    def get_state(self) -> List[np.float64]:
        """Get the current state of all floating-point registers"""
        return self.registers.copy()
    
    def reset(self) -> None:
        """Reset all floating-point registers to 0.0"""
        self.registers = [np.float64(0.0) for _ in range(32)] 