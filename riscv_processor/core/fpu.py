from enum import Enum
import numpy as np

class FPUOp(Enum):
    FLD = 0    # Load double
    FSD = 1    # Store double
    FMADD = 2  # Fused multiply-add
    FMSUB = 3  # Fused multiply-subtract
    FNMSUB = 4 # Negated fused multiply-subtract
    FNMADD = 5 # Negated fused multiply-add
    FADD = 6   # Add
    FSUB = 7   # Subtract
    FMUL = 8   # Multiply
    FDIV = 9   # Divide
    FMIN = 10  # Minimum
    FMAX = 11  # Maximum
    FLE = 12   # Less than or equal
    FLT = 13   # Less than
    FEQ = 14   # Equal
    FCVT_W_S = 15  # Convert float to signed integer (single precision)
    FCVT_W_D = 16  # Convert float to signed integer (double precision)
    FCVT_S_W = 17  # Convert signed integer to float (single precision)
    FCVT_D_W = 18  # Convert signed integer to float (double precision)

class FPU:
    def __init__(self):
        self.result: np.float64 = np.float64(0.0)
        self.exception: bool = False
        
    def execute(self, op: FPUOp, operand1: np.float64, operand2: np.float64 = 0.0, operand3: np.float64 = 0.0) -> np.float64:
        """Execute FPU operation"""
        try:
            if op == FPUOp.FLD:
                self.result = operand1  # Just pass through the loaded value
            elif op == FPUOp.FSD:
                self.result = operand1  # Just pass through the value to store
            elif op == FPUOp.FMADD:
                self.result = np.float64(operand1 * operand2 + operand3)
            elif op == FPUOp.FMSUB:
                self.result = np.float64(operand1 * operand2 - operand3)
            elif op == FPUOp.FNMSUB:
                self.result = np.float64(-(operand1 * operand2) - operand3)
            elif op == FPUOp.FNMADD:
                self.result = np.float64(-(operand1 * operand2) + operand3)
            elif op == FPUOp.FADD:
                self.result = np.float64(operand1 + operand2)
            elif op == FPUOp.FSUB:
                self.result = np.float64(operand1 - operand2)
            elif op == FPUOp.FMUL:
                self.result = np.float64(operand1 * operand2)
            elif op == FPUOp.FDIV:
                if operand2 == 0.0:
                    raise ValueError("Division by zero")
                self.result = np.float64(operand1 / operand2)
            elif op == FPUOp.FMIN:
                self.result = np.float64(min(operand1, operand2))
            elif op == FPUOp.FMAX:
                self.result = np.float64(max(operand1, operand2))
            elif op == FPUOp.FLE:
                self.result = np.float64(1.0 if operand1 <= operand2 else 0.0)
            elif op == FPUOp.FLT:
                self.result = np.float64(1.0 if operand1 < operand2 else 0.0)
            elif op == FPUOp.FEQ:
                self.result = np.float64(1.0 if operand1 == operand2 else 0.0)
            elif op in [FPUOp.FCVT_W_S, FPUOp.FCVT_W_D]:
                # Convert float to signed integer
                self.result = np.float64(np.int64(np.round(operand1)))
            elif op in [FPUOp.FCVT_S_W, FPUOp.FCVT_D_W]:
                # Convert signed integer to float
                self.result = np.float64(np.int64(operand1))
            
            self.exception = False
            return self.result
            
        except (ValueError, OverflowError, FloatingPointError) as e:
            self.exception = True
            raise e
    
    def get_result(self) -> np.float64:
        """Get the last FPU result"""
        return self.result
    
    def has_exception(self) -> bool:
        """Check if the last operation resulted in an exception"""
        return self.exception
    
    @staticmethod
    def is_nan(value: np.float64) -> bool:
        """Check if a value is NaN"""
        return np.isnan(value)
    
    @staticmethod
    def is_inf(value: np.float64) -> bool:
        """Check if a value is infinity"""
        return np.isinf(value)
    
    @staticmethod
    def is_zero(value: np.float64) -> bool:
        """Check if a value is zero"""
        return value == 0.0 