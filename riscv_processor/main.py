from riscv_processor.core.pipeline import Pipeline
import numpy as np

def create_sample_program() -> bytes:
    """Create a comprehensive RISC-V program that tests all instruction types and hazard types"""
    instructions = [
        # LUI x1, 0x1
        0x00001037,  # Load Upper Immediate: x1 = 0x1000
        
        # ADDI x2, x1, 100
        0x06408113,  # Add Immediate: x2 = x1 + 100
        
        # LW x3, 0(x2)
        0x00012183,  # Load Word: x3 = Memory[x2 + 0]
        
        # MUL x4, x3, x2
        0x02218233,  # Multiply: x4 = x3 * x2
        
        # DIV x5, x4, x3
        0x023242b3,  # Divide: x5 = x4 / x3
        
        # ADD x6, x5, x1
        0x001282b3,  # Add: x6 = x5 + x1
        
        # BEQ x6, x0, label
        0x00030463,  # Branch if Equal: if(x6 == x0) goto label
        
        # ADD x7, x6, x6
        0x006303b3,  # Add: x7 = x6 + x6
        
        # FLD f1, 0(x2)
        0x00013087,  # Float Load Double: f1 = Memory[x2 + 0]
        
        # FADD f2, f1, f1
        0x001081d3,  # Float Add: f2 = f1 + f1
        
        # FDIV f3, f2, f1
        0x0020a253,  # Float Divide: f3 = f2 / f1
        
        # FMUL f4, f3, f2
        0x002182d3,  # Float Multiply: f4 = f3 * f2
        
        # JAL x0, end
        0x008001ef,  # Jump and Link: Jump to end
        
        # FADD f5, f4, f4
        0x004203d3,  # Float Add: f5 = f4 + f4
        
        # SW x6, 0(x2)
        0x00612023,  # Store Word: Memory[x2 + 0] = x6
    ]
    
    # Convert instructions to bytes
    program = bytearray()
    for instr in instructions:
        # Ensure each instruction is written as a 32-bit little-endian value
        program.extend(instr.to_bytes(4, byteorder='little'))
    
    return bytes(program)

def main():
    # Create and initialize the pipeline
    pipeline = Pipeline()
    
    # Load the sample program
    program = create_sample_program()
    pipeline.load_program(program)
    
    print("Starting RISC-V Pipeline Simulation")
    print("==================================")
    
    # Execute the program step by step
    step = 1
    while True:
        try:
            state = pipeline.step()
            if state is None:
                break
                
            print(f"\nStep {step}:")
            print(f"PC: 0x{state.pc:08x}")
            # Ensure the instruction is displayed as a 32-bit value
            instr = state.instruction & 0xFFFFFFFF
            print(f"Instruction: 0x{instr:08x}")
            print(f"Control Signals: {state.control_signals}")
            print(f"RS1 Value: {state.rs1_value}")
            print(f"RS2 Value: {state.rs2_value}")
            print(f"ALU Result: {state.get_signed_alu_result()} (unsigned: {state.alu_result})")
            print(f"Memory Data: {state.memory_data}")
            print(f"Write Data: {state.write_data}")
            print(f"Stall: {state.stall}")
            print(f"Flush: {state.flush}")
            
            step += 1
            
            # Stop after 30 steps to prevent infinite loops
            if step > 30:
                break
        except ValueError as e:
            print(f"\nProgram terminated: {str(e)}")
            break
    
    print("\nFinal Register State:")
    for i, value in enumerate(pipeline.register_file.get_state()):
        # Convert to signed value for display
        signed_value = np.int64(value)
        print(f"x{i}: {signed_value} (unsigned: {value})")
    
    print("\nFinal FPU Register State:")
    for i, value in enumerate(pipeline.fpu_register_file.get_state()):
        print(f"f{i}: {value}")
    
    # Print final memory state
    print("\nFinal Memory State:")
    print("Program Memory:")
    for addr in range(0, 0x40, 4):  # Display first 64 bytes of program memory
        if addr in pipeline.memory.memory:
            print(f"0x{addr:08x}: 0x{pipeline.memory.read_word(addr):08x}")
    
    print("\nData Memory:")
    for addr in range(pipeline.data_memory_base, pipeline.data_memory_base + 0x100, 8):  # Display first 256 bytes of data memory
        if addr in pipeline.memory.memory:
            value = pipeline.memory.read_double(addr)
            print(f"0x{addr:08x}: {value}")

if __name__ == "__main__":
    main() 