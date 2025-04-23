from typing import Dict, Optional
import numpy as np

class Memory:
    def __init__(self, size: int = 1024 * 1024):  # Default 1MB memory
        self.memory: Dict[int, np.uint8] = {}
        # Ensure size is a power of 2 and at least 4KB
        if size < 4096:
            size = 4096
        # Round up to the next power of 2
        size = 1 << (size - 1).bit_length()
        # Create an address mask for fast bounds checking
        self.size = size
        self.address_mask = size - 1
        
    def read_byte(self, address: int) -> np.uint8:
        """Read a byte from memory"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        return self.memory.get(address, np.uint8(0))
    
    def write_byte(self, address: int, value: np.uint8) -> None:
        """Write a byte to memory"""
        # Convert address to unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        self.memory[address] = value
    
    def read_word(self, address: int) -> np.uint32:
        """Read a 32-bit word from memory (little-endian)"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        # Ensure the address is word-aligned
        if address & 0x3:
            raise ValueError(f"Unaligned memory access at address: {address}")
        value = 0
        for i in range(4):
            value |= int(self.read_byte(address + i)) << (i * 8)
        return np.uint32(value)
    
    def write_word(self, address: int, value: np.uint32) -> None:
        """Write a 32-bit word to memory (little-endian)"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        # Ensure the address is word-aligned
        if address & 0x3:
            raise ValueError(f"Unaligned memory access at address: {address}")
        for i in range(4):
            self.write_byte(address + i, np.uint8((value >> (i * 8)) & 0xFF))
    
    def load_program(self, program: bytes, start_address: int = 0) -> None:
        """Load a program into memory starting at the specified address"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        start_address = (start_address & 0xFFFFFFFF) & self.address_mask
        # Ensure the program fits in memory
        if start_address + len(program) > self.size:
            raise ValueError("Program size exceeds available memory")
        for i, byte in enumerate(program):
            self.write_byte(start_address + i, np.uint8(byte))
    
    def clear(self) -> None:
        """Clear all memory"""
        self.memory.clear()
    
    def read_double(self, address: int) -> np.float64:
        """Read a 64-bit double-precision floating-point value from memory (little-endian)"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        # Ensure the address is double-word-aligned
        if address & 0x7:
            raise ValueError(f"Unaligned memory access at address: {address}")
        
        # Read 8 bytes as a 64-bit integer
        value = 0
        for i in range(8):
            value |= int(self.read_byte(address + i)) << (i * 8)
        
        # Convert bytes to float64 using numpy's view method
        return np.frombuffer(np.array([value], dtype=np.uint64).tobytes(), dtype=np.float64)[0]
    
    def write_double(self, address: int, value: np.float64) -> None:
        """Write a 64-bit double-precision floating-point value to memory (little-endian)"""
        # Convert address to an unsigned 32-bit value and mask to valid range
        address = (address & 0xFFFFFFFF) & self.address_mask
        # Ensure the address is double-word-aligned
        if address & 0x7:
            raise ValueError(f"Unaligned memory access at address: {address}")
        
        # Convert float64 to bytes
        bytes_value = np.array([value], dtype=np.float64).tobytes()
        
        # Write 8 bytes
        for i in range(8):
            self.write_byte(address + i, np.uint8(bytes_value[i])) 