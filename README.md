# RISC-V Pipelined Processor Implementation

This project is a structural simulation of a **pipelined RISC-V processor** supporting the **RV64IM** base and **D-extension** (double-precision floating-point), built with hazard detection and forwarding units.

---

## 📝 Project Statement  
**Design and implement a pipelined RISC-V processor supporting RV64IM with D-extension, featuring hazard detection and forwarding units.**

---

## 👥 Contributors  
- **Jashandeep Singh** (IMT2024022)  
- **Heer** (IMT2024031)

---

## Features
- 64-bit RISC-V processor implementation
- Support for RV64IM (Integer and Multiplication/Division) and D (Double-precision floating-point) extensions
- 5-stage pipeline implementation:
  - Instruction Fetch (IF)
  - Instruction Decode (ID)
  - Execute (EX)
  - Memory Access (MEM)
  - Write Back (WB)
- Hazard detection and forwarding units
- IEEE 754 floating-point implementation
- Structural implementation of arithmetic units

## Project Structure
```
riscv_processor/
├── core/
│   ├── __init__.py
│   ├── registers.py
│   ├── memory.py
│   ├── alu.py
│   ├── fpu.py
│   ├── control.py
│   └── pipeline.py
├── utils/
│   ├── __init__.py
│   ├── instruction_decoder.py
│   └── hazard_detector.py
└── main.py
```

## Setup
1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the processor with:
```bash
python main.py
```

## Implementation Details
- Register file: 64-bit general-purpose registers (x0-x31)
- Memory: 64-bit address space
- ALU: Supports all RV64I operations
- FPU: IEEE 754 double-precision floating-point operations
- Pipeline stages with hazard detection and forwarding 
