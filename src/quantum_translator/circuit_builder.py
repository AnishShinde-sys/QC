from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Dict, Any, Optional, List
import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit.transpiler.passes import CommutativeCancellation, Depth
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class QubitAllocator:
    """Manages qubit allocation and tracking"""
    
    def __init__(self):
        self.variable_to_qubit = {}  # Maps variable names to qubit indices
        self.next_qubit = 0
    
    def allocate_qubit(self, var_name: str) -> int:
        """Allocate a new qubit for a variable"""
        if var_name in self.variable_to_qubit:
            return self.variable_to_qubit[var_name]
        
        qubit_idx = self.next_qubit
        self.variable_to_qubit[var_name] = qubit_idx
        self.next_qubit += 1
        return qubit_idx
    
    def get_qubit(self, var_name: str) -> Optional[int]:
        """Get qubit index for a variable"""
        return self.variable_to_qubit.get(var_name)
    
    def total_qubits(self) -> int:
        """Get total number of allocated qubits"""
        return self.next_qubit

class CircuitOptimizer:
    """Handles quantum circuit optimization"""
    
    def __init__(self):
        self.pass_manager = PassManager()
        self._setup_optimization_passes()
        self.debug = True  # Enable debugging
    
    def _debug(self, message: str):
        """Print debug information if debug mode is enabled"""
        if self.debug and logger:
            logger.debug(message)
    
    def _setup_optimization_passes(self):
        """Configure optimization passes"""
        from qiskit.transpiler.passes import (
            Optimize1qGates,
            CXCancellation,
            CommutativeCancellation,
            Depth,
            RemoveResetInZeroState,
            OptimizeSwapBeforeMeasure,
            RemoveDiagonalGatesBeforeMeasure
        )
        
        self.pass_manager.append([
            Optimize1qGates(),        # Merge single-qubit gates
            CXCancellation(),         # Cancel adjacent CNOT gates
            CommutativeCancellation(),# Cancel commuting gates
            RemoveResetInZeroState(), # Remove unnecessary resets
            OptimizeSwapBeforeMeasure(), # Optimize swap gates
            RemoveDiagonalGatesBeforeMeasure(), # Remove unnecessary diagonal gates
            Depth()                   # Reduce circuit depth
        ])
    
    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize the quantum circuit"""
        try:
            if self.debug:
                self._debug(f"Pre-optimization depth: {circuit.depth()}")
            optimized = self.pass_manager.run(circuit)
            if self.debug:
                self._debug(f"Post-optimization depth: {optimized.depth()}")
            return optimized
        except Exception as e:
            self._debug(f"Optimization failed: {str(e)}")
            return circuit  # Return original circuit if optimization fails

class FunctionContext:
    """Manages function context and variable scope"""
    
    def __init__(self, name: str, args: Dict[str, Any]):
        self.name = name
        self.args = args
        self.return_value = None
        self.local_variables = set()

class QuantumCircuitError(Exception):
    """Base exception for quantum circuit errors"""
    pass

class QubitAllocationError(QuantumCircuitError):
    """Error in qubit allocation"""
    pass

class OperationError(QuantumCircuitError):
    """Error in quantum operations"""
    pass

class CircuitBuildError(QuantumCircuitError):
    """Error in circuit building"""
    pass

def error_context(func):
    """Decorator to add error context"""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self._debug(f"Error in {func.__name__}: {str(e)}")
            if isinstance(e, QuantumCircuitError):
                raise
            raise QuantumCircuitError(f"Error in {func.__name__}: {str(e)}")
    return wrapper

class QuantumCircuitBuilder:
    """Builds quantum circuits from intermediate representation"""
    
    def __init__(self):
        """Initialize the circuit builder"""
        self.allocator = QubitAllocator()
        self.qubit_mapping = {}  # Maps variable names to qubit indices
        self.circuit = None
        self.debug_level = 1
        self.debug_stats = {
            'operations': 0,
            'allocations': 0,
            'errors': 0
        }
        self.measurement_map = {}  # Track which qubits need measurement
        self.optimizer = CircuitOptimizer()
        self.functions = {}  # Store function definitions
        self.current_function = None  # Track current function context
        self.debug = True  # Enable/disable debug output
        self.debug_level = 1  # 0: None, 1: Basic, 2: Detailed
        self.debug_stats = {'operations': 0, 'allocations': 0, 'errors': 0}
    
    def _debug(self, message: str, level: int = 1):
        """Enhanced debug output with levels"""
        if self.debug and self.debug_level >= level:
            # Add timestamp and context
            context = f"[{self.current_function.name if self.current_function else 'main'}]"
            logger.debug(f"{context} {message}")
    
    def build_circuit(self, ir: Dict[str, Any]) -> QuantumCircuit:
        """Convert IR to an optimized quantum circuit with debugging"""
        try:
            self._debug("Starting circuit build")
            
            # Create basic circuit
            circuit = self._build_basic_circuit(ir)
            self._debug_circuit_stats("Basic circuit", circuit)
            
            # Optimize the circuit
            self._debug("Optimizing circuit")
            optimized_circuit = self.optimizer.optimize(circuit)
            self._debug_circuit_stats("Optimized circuit", optimized_circuit)
            
            # Add measurements
            self._debug("Adding measurements")
            self.add_measurements(optimized_circuit)
            self._debug_circuit_stats("Final circuit", optimized_circuit)
            
            return optimized_circuit
            
        except Exception as e:
            self._debug(f"Circuit build failed: {str(e)}")
            raise QuantumCircuitError(f"Circuit build failed: {str(e)}")
    
    def _debug_circuit_stats(self, stage: str, circuit: QuantumCircuit):
        """Enhanced circuit statistics"""
        if not self.debug:
            return
            
        stats = {
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'size': circuit.size(),
            'num_nonlocal_gates': len([g for g in circuit.data if len(g[1]) > 1]),
            'num_measurements': len([g for g in circuit.data if g[0].name == 'measure']),
            'gate_counts': self._count_gates(circuit),
            'qubit_usage': self._analyze_qubit_usage()
        }
        
        self._debug(f"\n{stage} statistics:", level=1)
        for key, value in stats.items():
            self._debug(f"  {key}: {value}", level=2)
    
    def _count_gates(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Count different types of gates"""
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction[0].name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        return gate_counts
    
    def _analyze_qubit_usage(self) -> Dict[str, List[int]]:
        """Analyze how qubits are being used"""
        usage = {
            'variables': [],
            'temporaries': [],
            'controls': [],
            'unused': []
        }
        
        for var, qubit in self.allocator.variable_to_qubit.items():
            if var.startswith('temp_'):
                usage['temporaries'].append(qubit)
            elif var.startswith(('if_control_', 'loop_control_')):
                usage['controls'].append(qubit)
            else:
                usage['variables'].append(qubit)
        
        return usage
    
    def _build_basic_circuit(self, ir: Dict[str, Any]) -> QuantumCircuit:
        """Build the basic circuit before optimization"""
        # Calculate initial circuit size
        required_qubits = max(1, self.allocator.total_qubits())
        circuit_size = required_qubits + 100  # Add buffer for temporary qubits
        
        # Create registers with sufficient size
        qr = QuantumRegister(circuit_size, 'q')
        cr = ClassicalRegister(circuit_size, 'c')
        self.circuit = QuantumCircuit(qr, cr)
        
        # Initialize all qubits to |0⟩
        for i in range(circuit_size):
            self.circuit.reset(i)
        
        try:
            # Pre-analyze to determine qubit requirements
            self._analyze_qubit_requirements(ir)
            
            # Process the IR nodes
            self._process_ir_nodes(self.circuit, ir)
        except QuantumCircuitError as e:
            if "Index out of range" in str(e):
                # If we run out of qubits, try again with larger circuit
                return self._build_basic_circuit_with_size(ir, circuit_size * 2)
            raise e
        except Exception as e:
            raise QuantumCircuitError(f"Circuit build failed: {str(e)}")
        
        return self.circuit
    
    def _build_basic_circuit_with_size(self, ir: Dict[str, Any], size: int) -> QuantumCircuit:
        """Build circuit with specified size"""
        # Create registers with sufficient size
        qr = QuantumRegister(size, 'q')
        cr = ClassicalRegister(size, 'c')
        self.circuit = QuantumCircuit(qr, cr)
        
        # Initialize all qubits to |0⟩
        for i in range(size):
            self.circuit.reset(i)
        
        try:
            # Pre-analyze to determine qubit requirements
            self._analyze_qubit_requirements(ir)
            
            # Process the IR nodes
            self._process_ir_nodes(self.circuit, ir)
        except Exception as e:
            if "Index out of range" in str(e):
                # If still not enough qubits, try again with even larger circuit
                return self._build_basic_circuit_with_size(ir, size * 2)
            raise QuantumCircuitError(f"Circuit build failed: {str(e)}")
        
        return self.circuit
    
    def _analyze_qubit_requirements(self, ir: Dict[str, Any]) -> None:
        """Pre-analyze IR to determine qubit requirements"""
        if ir['type'] == 'module':
            for node in ir['body']:
                self._analyze_qubit_requirements(node)
        elif ir['type'] == 'assign':
            # Allocate target qubit
            target = ir['targets'][0]
            if target['type'] == 'name':
                if target['id'] not in self.qubit_mapping:
                    self._allocate_qubit(target['id'])
            
            # Analyze value expression
            if 'value' in ir:
                self._analyze_expression_requirements(ir['value'])
        elif ir['type'] == 'if':
            # Allocate control qubit
            control_name = f"if_control_{len(self.qubit_mapping)}"
            control_qubit = self._allocate_qubit(control_name)
            ir['control_qubit'] = control_qubit
            
            # Analyze condition
            self._analyze_expression_requirements(ir['test'])
            
            # Analyze both branches
            for node in ir['body']:
                self._analyze_qubit_requirements(node)
            for node in ir['orelse']:
                self._analyze_qubit_requirements(node)
        elif ir['type'] == 'for':
            # Allocate loop variable and control
            if ir['target']['type'] == 'name':
                loop_var = ir['target']['id']
                if loop_var not in self.qubit_mapping:
                    self._allocate_qubit(loop_var)
                control_name = f"loop_control_{loop_var}"
                control_qubit = self._allocate_qubit(control_name)
                ir['control_qubit'] = control_qubit
            
            # Analyze loop body
            for node in ir['body']:
                self._analyze_qubit_requirements(node)
    
    def _process_ir_nodes(self, circuit: QuantumCircuit, ir: Dict[str, Any], control: Optional[int] = None):
        """Process IR nodes with optional control qubit"""
        try:
            if ir['type'] == 'module':
                for node in ir['body']:
                    self._process_ir_nodes(circuit, node, control)
            elif ir['type'] == 'assign':
                self._process_assignment(circuit, ir, control)
            elif ir['type'] == 'if':
                self._process_conditional(circuit, ir)
            elif ir['type'] == 'for':
                self._process_range_loop(circuit, ir)
            elif ir['type'] == 'break':
                return True  # Signal break to parent loop
            else:
                raise QuantumCircuitError(f"Unsupported node type: {ir['type']}")
                
            return False  # No break encountered
            
        except Exception as e:
            raise QuantumCircuitError(f"Error processing node {ir['type']}: {str(e)}")
    
    def _process_assignment(self, circuit: QuantumCircuit, ir: Dict[str, Any], control: Optional[int] = None):
        """Process assignment with optional control"""
        try:
            # Get target qubit
            target = ir['targets'][0]  # Currently only supporting single target
            target_qubit = self.allocator.get_qubit(target['id'])
            
            if target_qubit is None:
                target_qubit = self.allocator.allocate_qubit(target['id'])
            
            # Get value qubit
            value_qubit = self._get_operand_qubit(circuit, ir['value'])
            
            # If controlled, use CCNOT (Toffoli) gate
            if control is not None:
                circuit.ccx(control, value_qubit, target_qubit)
            else:
                # Regular assignment
                circuit.cx(value_qubit, target_qubit)
                
        except Exception as e:
            raise QuantumCircuitError(f"Assignment failed: {str(e)}")
    
    def _assign_constant(self, circuit: QuantumCircuit, target_qubit: int, value: int):
        """Assign a classical constant to a qubit"""
        # Reset qubit to |0⟩ state
        circuit.reset(target_qubit)
        
        # Convert value to binary (use least significant bit)
        if value % 2 == 1:
            circuit.x(target_qubit)
    
    def _copy_qubit(self, circuit: QuantumCircuit, source: int, target: int):
        """Copy value from source qubit to target qubit"""
        circuit.reset(target)
        circuit.cx(source, target)  # CNOT gate for copying
    
    @error_context
    def _process_binary_operation(self, circuit: QuantumCircuit, ir: Dict[str, Any]) -> int:
        """Process binary operations with enhanced tracking"""
        try:
            op = ir['op']
            self._debug(f"Starting binary operation: {op}", level=2)
            self.debug_stats['operations'] += 1
            
            # Get operands
            left = self._get_operand_qubit(circuit, ir['left'])
            right = self._get_operand_qubit(circuit, ir['right'])
            
            if left is None or right is None:
                raise OperationError(f"Invalid operands for {op}")
            
            self._debug(f"Operands: left={left}, right={right}", level=2)
            
            # Allocate result qubit
            result = self.allocator.allocate_qubit(f"temp_binop_{self.allocator.total_qubits()}")
            
            # Apply operation based on operator type
            if op == 'add':
                self._quantum_addition(circuit, left, right, result)
            elif op == 'sub':
                self._quantum_subtraction(circuit, left, right, result)
            elif op == 'mult':
                self._quantum_multiplication(circuit, left, right, result)
            elif op == 'and':
                self._quantum_bitwise_and(circuit, left, right, result)
            elif op == 'or':
                self._quantum_bitwise_or(circuit, left, right, result)
            elif op == 'xor':
                self._quantum_bitwise_xor(circuit, left, right, result)
            else:
                raise OperationError(f"Unsupported binary operator: {op}")
            
            self._debug(f"Operation {op} completed. Result qubit: {result}", level=2)
            return result
            
        except Exception as e:
            self.debug_stats['errors'] += 1
            raise OperationError(f"Binary operation failed: {str(e)}")
    
    def _quantum_addition(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum addition using CNOT and Toffoli gates"""
        # Initialize result qubit to |0⟩
        circuit.reset(result)
        
        # Use CNOT gates to add qubits
        circuit.cx(a, result)  # Add first number
        circuit.cx(b, result)  # Add second number
        
        # Add carry handling for multi-qubit addition
        carry = self.allocator.allocate_qubit(f"carry_{self.allocator.total_qubits()}")
        circuit.reset(carry)
        circuit.ccx(a, b, carry)  # Toffoli gate for carry
    
    @error_context
    def _get_operand_qubit(self, circuit: QuantumCircuit, operand: Dict[str, Any]) -> int:
        """Get operand qubit with enhanced tracking"""
        try:
            if operand['type'] == 'name':
                qubit = self.allocator.get_qubit(operand['id'])
                if qubit is None:
                    raise QubitAllocationError(f"Variable '{operand['id']}' not allocated")
                return qubit
            elif operand['type'] == 'constant':
                qubit = self.allocator.allocate_qubit(f"const_{operand['value']}")
                self._assign_constant(circuit, qubit, operand['value'])
                return qubit
            elif operand['type'] == 'binop':
                # Process binary operation and return its result qubit
                left_qubit = self._get_operand_qubit(circuit, operand['left'])
                right_qubit = self._get_operand_qubit(circuit, operand['right'])
                
                # Allocate result qubit after getting operands
                result_qubit = self.allocator.allocate_qubit(f"temp_binop_{self.allocator.total_qubits()}")
                
                # Apply operation based on operator type
                if operand['op'] == 'add':
                    self._quantum_addition(circuit, left_qubit, right_qubit, result_qubit)
                elif operand['op'] == 'sub':
                    self._quantum_subtraction(circuit, left_qubit, right_qubit, result_qubit)
                elif operand['op'] == 'mult':
                    self._quantum_multiplication(circuit, left_qubit, right_qubit, result_qubit)
                elif operand['op'] == 'and':
                    self._quantum_bitwise_and(circuit, left_qubit, right_qubit, result_qubit)
                elif operand['op'] == 'or':
                    self._quantum_bitwise_or(circuit, left_qubit, right_qubit, result_qubit)
                elif operand['op'] == 'xor':
                    self._quantum_bitwise_xor(circuit, left_qubit, right_qubit, result_qubit)
                else:
                    raise QubitAllocationError(f"Unsupported binary operator: {operand['op']}")
                
                return result_qubit
            elif operand['type'] == 'compare':
                # Handle comparison operations
                left_qubit = self._get_operand_qubit(circuit, operand['left'])
                right_qubit = self._get_operand_qubit(circuit, operand['comparators'][0])
                
                # Allocate result qubit after getting operands
                result_qubit = self.allocator.allocate_qubit(f"temp_compare_{self.allocator.total_qubits()}")
                
                # Implement comparison
                circuit.reset(result_qubit)
                if operand['ops'][0] == 'eq':
                    circuit.cx(left_qubit, result_qubit)
                    circuit.cx(right_qubit, result_qubit)
                    circuit.x(result_qubit)  # XNOR for equality
                elif operand['ops'][0] == 'gt':
                    # Simple implementation - assumes binary values
                    circuit.cx(left_qubit, result_qubit)
                    circuit.x(right_qubit)
                    circuit.ccx(left_qubit, right_qubit, result_qubit)
                    circuit.x(right_qubit)  # Restore right operand
                
                return result_qubit
                
            raise QubitAllocationError(f"Unsupported operand type: {operand['type']}")
            
        except Exception as e:
            raise QubitAllocationError(f"Qubit allocation failed: {str(e)}")
    
    def add_measurements(self, circuit: QuantumCircuit):
        """Add measurement operations to the circuit"""
        for var, qubit in self.allocator.variable_to_qubit.items():
            if not var.startswith(('temp_', 'carry_', 'const_')):
                circuit.measure(qubit, qubit)
                self.measurement_map[var] = qubit
    
    def _quantum_multiplication(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum multiplication using repeated addition"""
        # Initialize result qubit to |0⟩
        circuit.reset(result)
        
        # Use ancilla qubit for temporary storage
        temp = self.allocator.allocate_qubit(f"temp_mult_{self.allocator.total_qubits()}")
        circuit.reset(temp)
        
        # Controlled addition: if b is 1, add a to result
        circuit.ccx(b, a, temp)  # AND gate
        circuit.cx(temp, result)  # Add to result if temp is 1
        circuit.reset(temp)
    
    def _quantum_division(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum division using quantum Fourier transform"""
        # Allocate ancilla qubits for QFT
        n_qubits = 4  # Number of qubits for precision
        ancilla = [self.allocator.allocate_qubit(f"div_ancilla_{i}") 
                  for i in range(n_qubits)]
        
        # Apply QFT
        self._apply_qft(circuit, ancilla)
        
        # Controlled rotations based on divisor
        for i, q in enumerate(ancilla):
            angle = 2 * np.pi / (2 ** (i + 1))
            circuit.cp(angle, b, q)
        
        # Inverse QFT
        self._apply_inverse_qft(circuit, ancilla)
        
        # Copy result
        circuit.cx(ancilla[-1], result)
        
        # Clean up ancilla qubits
        for q in ancilla:
            circuit.reset(q)
    
    def _apply_qft(self, circuit: QuantumCircuit, qubits: List[int]):
        """Apply Quantum Fourier Transform"""
        for i, q in enumerate(qubits):
            circuit.h(q)
            for j, r in enumerate(qubits[i+1:], i+1):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                circuit.cp(angle, q, r)
    
    def _apply_inverse_qft(self, circuit: QuantumCircuit, qubits: List[int]):
        """Apply inverse Quantum Fourier Transform"""
        for i in reversed(range(len(qubits))):
            for j in reversed(range(i+1, len(qubits))):
                angle = -2 * np.pi / (2 ** (j - i + 1))
                circuit.cp(angle, qubits[i], qubits[j])
            circuit.h(qubits[i])
    
    def _process_loop(self, circuit: QuantumCircuit, ir: Dict[str, Any]):
        """Process for loop constructs"""
        if ir['iter']['type'] == 'range':
            self._process_range_loop(circuit, ir)
        else:
            raise ValueError("Only range-based loops are supported")
    
    def _process_range_loop(self, circuit: QuantumCircuit, ir: Dict[str, Any]):
        """Process range-based for loops"""
        try:
            args = ir['iter']['args']
            
            # Parse range arguments
            if len(args) == 1:
                start, stop, step = 0, args[0]['value'], 1
            elif len(args) == 2:
                start = args[0]['value']
                stop = args[1]['value']
                step = 1
            else:
                start = args[0]['value']
                stop = args[1]['value']
                step = args[2]['value']
            
            # Validate range
            if stop < start and step > 0:
                return  # Empty range
            if step == 0:
                raise ValueError("Range step cannot be zero")
            
            # Get loop variable qubit
            loop_var = ir['target']['id']
            loop_qubit = self.allocator.get_qubit(loop_var)
            control_qubit = ir.get('control_qubit')
            
            if loop_qubit is None or control_qubit is None:
                raise QuantumCircuitError("Loop qubits not properly allocated")
            
            # Initialize qubits
            circuit.reset(loop_qubit)
            circuit.reset(control_qubit)
            
            # Process loop iterations
            for i in range(start, stop, step):
                # Set loop variable
                self._assign_constant(circuit, loop_qubit, i)
                
                # Process body with control
                for node in ir['body']:
                    if node.get('type') == 'break':
                        return  # Exit loop on break
                    self._process_ir_nodes(circuit, node, control=control_qubit)
                
                # Reset control for next iteration
                circuit.reset(control_qubit)
                
        except Exception as e:
            raise QuantumCircuitError(f"Error processing loop: {str(e)}")
    
    def _create_loop_superposition(self, circuit: QuantumCircuit, qubit: int, 
                                 start: int, stop: int, step: int):
        """Create superposition of loop variable values"""
        # Reset qubit to |0⟩
        circuit.reset(qubit)
        
        # Apply Hadamard to create initial superposition
        circuit.h(qubit)
        
        # Apply phase rotations for each value in range
        num_iterations = (stop - start) // step
        for i in range(num_iterations):
            angle = 2 * np.pi * i / num_iterations
            circuit.p(angle, qubit)
    
    def _quantum_modulo(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum modulo operation"""
        # Allocate ancilla qubits
        temp = self.allocator.allocate_qubit(f"mod_temp_{self.allocator.total_qubits()}")
        circuit.reset(temp)
        
        # First perform division
        self._quantum_division(circuit, a, b, temp)
        
        # Multiply quotient by divisor
        mult_result = self.allocator.allocate_qubit(f"mod_mult_{self.allocator.total_qubits()}")
        self._quantum_multiplication(circuit, temp, b, mult_result)
        
        # Subtract from original number to get remainder
        circuit.reset(result)
        circuit.cx(a, result)  # Copy a to result
        circuit.cx(mult_result, result)  # Subtract multiplication result
        
        # Clean up
        circuit.reset(temp)
        circuit.reset(mult_result)
    
    def _quantum_exponentiation(self, circuit: QuantumCircuit, base: int, exponent: int, result: int):
        """Implement quantum exponentiation using controlled rotations"""
        # Reset result qubit
        circuit.reset(result)
        
        # Apply controlled rotation gates
        angle = 2 * np.pi / (2 ** exponent)
        for _ in range(2 ** exponent):
            circuit.cp(angle, base, result)
    
    def _quantum_bitwise_and(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum AND operation"""
        circuit.reset(result)
        circuit.ccx(a, b, result)  # Toffoli gate implements AND
    
    def _quantum_bitwise_or(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum OR operation"""
        circuit.reset(result)
        
        # OR can be implemented using AND and NOT
        temp = self.allocator.allocate_qubit(f"or_temp_{self.allocator.total_qubits()}")
        circuit.reset(temp)
        
        # NOT(AND(NOT(a), NOT(b)))
        circuit.x(a)  # NOT a
        circuit.x(b)  # NOT b
        circuit.ccx(a, b, temp)  # AND
        circuit.x(temp)  # NOT
        circuit.cx(temp, result)  # Copy to result
        
        # Clean up
        circuit.x(a)  # Restore a
        circuit.x(b)  # Restore b
        circuit.reset(temp)
    
    def _quantum_bitwise_xor(self, circuit: QuantumCircuit, a: int, b: int, result: int):
        """Implement quantum XOR operation"""
        circuit.reset(result)
        circuit.cx(a, result)  # CNOT implements XOR
        circuit.cx(b, result)
    
    def _process_function_definition(self, ir: Dict[str, Any]):
        """Store function definition for later use"""
        function_name = ir['name']
        self.functions[function_name] = {
            'args': ir['args'],
            'body': ir['body'],
            'returns': ir['returns']
        }
    
    def _process_function_call(self, circuit: QuantumCircuit, ir: Dict[str, Any]):
        """Process function calls and apply quantum operations"""
        function_name = ir['function']
        if function_name not in self.functions:
            raise ValueError(f"Undefined function: {function_name}")
        
        # Create new function context
        func_def = self.functions[function_name]
        context = FunctionContext(function_name, ir['args'])
        prev_context = self.current_function
        self.current_function = context
        
        try:
            # Allocate qubits for arguments
            self._allocate_function_args(circuit, func_def['args'], ir['args'])
            
            # Process function body
            for node in func_def['body']:
                self._process_ir_nodes(circuit, node)
            
            # Handle return value
            if func_def['returns']:
                self._process_return(circuit, context)
                
        finally:
            # Restore previous context
            self.current_function = prev_context
    
    def _allocate_function_args(self, circuit: QuantumCircuit, 
                              func_args: Dict[str, Any], call_args: List[Dict[str, Any]]):
        """Allocate qubits for function arguments"""
        for param, arg in zip(func_args['args'], call_args):
            param_name = param['id']
            qubit = self._get_operand_qubit(circuit, arg)
            self.current_function.local_variables.add(param_name)
            self.allocator.variable_to_qubit[param_name] = qubit
    
    def _process_return(self, circuit: QuantumCircuit, context: FunctionContext):
        """Process function return value"""
        if context.return_value:
            return_qubit = self._get_operand_qubit(circuit, context.return_value)
            # Allocate a new qubit for the return value
            result_qubit = self.allocator.allocate_qubit(f"return_{context.name}")
            self._copy_qubit(circuit, return_qubit, result_qubit)
    
    def _process_conditional(self, circuit: QuantumCircuit, ir: Dict[str, Any]):
        """Process if statements"""
        try:
            # Get condition qubit
            condition = self._get_operand_qubit(circuit, ir['test'])
            
            # Get pre-allocated control qubit
            control = ir.get('control_qubit')
            if control is None:
                raise QuantumCircuitError("Control qubit not found")
            
            # Initialize control qubit
            circuit.reset(control)
            
            # Process true branch
            for node in ir['body']:
                self._process_ir_nodes(circuit, node)
                circuit.cx(condition, control)
            
            # Process false branch if it exists
            if ir['orelse']:
                circuit.x(condition)  # Invert condition
                for node in ir['orelse']:
                    self._process_ir_nodes(circuit, node)
                    circuit.cx(condition, control)
                circuit.x(condition)  # Restore condition
            
            # Clean up
            circuit.reset(control)
            
        except Exception as e:
            self._debug(f"Error in conditional processing: {str(e)}")
            raise QuantumCircuitError(f"Conditional processing failed: {str(e)}")
    
    def _ensure_qubit_allocation(self, circuit: QuantumCircuit, num_qubits: int) -> List[int]:
        """Ensure enough qubits are allocated and initialized"""
        qubits = []
        for i in range(num_qubits):
            qubit = self.allocator.allocate_qubit(f"temp_{self.allocator.total_qubits()}")
            circuit.reset(qubit)
            qubits.append(qubit)
        return qubits
    
    def _ensure_enough_qubits(self, circuit: QuantumCircuit, required_qubits: int):
        """Ensure circuit has enough qubits"""
        current_qubits = self.allocator.total_qubits()
        if current_qubits < required_qubits:
            # Extend quantum and classical registers
            qr = QuantumRegister(required_qubits - current_qubits, f'q{current_qubits}')
            cr = ClassicalRegister(required_qubits - current_qubits, f'c{current_qubits}')
            circuit.add_register(qr)
            circuit.add_register(cr)
    
    def _validate_qubit_indices(self, *qubits: int):
        """Validate qubit indices"""
        total_qubits = self.allocator.total_qubits()
        for qubit in qubits:
            if not isinstance(qubit, int):
                raise QuantumCircuitError(f"Invalid qubit index type: {type(qubit)}")
            if qubit < 0 or qubit >= total_qubits:
                raise QuantumCircuitError(f"Qubit index {qubit} out of range [0, {total_qubits})")
    
    def _print_status(self):
        """Print current status of the circuit building process"""
        if not self.debug:
            return
            
        self._debug("\nCurrent Status:", level=1)
        self._debug(f"  Operations performed: {self.debug_stats['operations']}", level=1)
        self._debug(f"  Qubit allocations: {self.debug_stats['allocations']}", level=1)
        self._debug(f"  Errors encountered: {self.debug_stats['errors']}", level=1)
        self._debug(f"  Total qubits: {self.allocator.total_qubits()}", level=1)
        
        if self.debug_level >= 2:
            self._debug("  Qubit mapping:", level=2)
            for var, qubit in self.allocator.variable_to_qubit.items():
                self._debug(f"    {var} → {qubit}", level=2)
    
    def _analyze_expression(self, expr: Dict[str, Any]) -> int:
        """Analyze an expression and return the qubit containing the result"""
        if expr['type'] == 'constant':
            # For constants, allocate a new qubit and set its value
            qubit = self._allocate_qubit()
            if expr['value'] == 1:
                self.circuit.x(qubit)
            return qubit
        
        elif expr['type'] == 'name':
            # For variables, return their assigned qubit
            if expr['id'] not in self.qubit_mapping:
                raise QuantumCircuitError(f"Variable {expr['id']} not defined")
            return self.qubit_mapping[expr['id']]
        
        elif expr['type'] == 'binop':
            # For binary operations, analyze operands and apply operation
            left_qubit = self._analyze_expression(expr['left'])
            right_qubit = self._analyze_expression(expr['right'])
            result_qubit = self._allocate_qubit()
            
            if expr['op'] == 'add':
                # Quantum addition
                self.circuit.cx(left_qubit, result_qubit)
                self.circuit.cx(right_qubit, result_qubit)
            elif expr['op'] == 'sub':
                # Quantum subtraction
                self.circuit.cx(left_qubit, result_qubit)
                self.circuit.x(right_qubit)
                self.circuit.cx(right_qubit, result_qubit)
                self.circuit.x(right_qubit)
            elif expr['op'] == 'mult':
                # Quantum multiplication (basic AND operation for binary values)
                self.circuit.ccx(left_qubit, right_qubit, result_qubit)
            elif expr['op'] == 'and':
                # Quantum AND
                self.circuit.ccx(left_qubit, right_qubit, result_qubit)
            elif expr['op'] == 'or':
                # Quantum OR
                self.circuit.cx(left_qubit, result_qubit)
                self.circuit.cx(right_qubit, result_qubit)
                self.circuit.ccx(left_qubit, right_qubit, result_qubit)
            else:
                raise QuantumCircuitError(f"Unsupported binary operation: {expr['op']}")
            
            return result_qubit
        
        else:
            raise QuantumCircuitError(f"Unsupported expression type: {expr['type']}")
    
    def _allocate_qubit(self, var_name: Optional[str] = None) -> int:
        """Allocate a new qubit and optionally map it to a variable name"""
        try:
            if not self.circuit:
                raise QuantumCircuitError("Circuit not initialized")
            
            if var_name:
                if var_name in self.qubit_mapping:
                    return self.qubit_mapping[var_name]
                qubit = self.allocator.allocate_qubit(var_name)
                self.qubit_mapping[var_name] = qubit
            else:
                qubit = self.allocator.allocate_qubit(f"temp_{self.allocator.total_qubits()}")
            
            # Verify qubit index is within circuit bounds
            if qubit >= self.circuit.num_qubits:
                raise IndexError(f"Index {qubit} out of range for size {self.circuit.num_qubits}")
            
            self.debug_stats['allocations'] += 1
            self._debug(f"Allocated qubit {qubit}" + (f" for {var_name}" if var_name else ""))
            return qubit
            
        except Exception as e:
            if isinstance(e, IndexError):
                raise QuantumCircuitError(f"Qubit allocation failed: {str(e)}")
            raise QuantumCircuitError(f"Qubit allocation failed: {str(e)}")
    
    def _analyze_expression_requirements(self, expr: Dict[str, Any]) -> None:
        """Analyze expression to determine required qubits"""
        if expr['type'] == 'constant':
            # Constants need a new qubit
            self._allocate_qubit(f"const_{expr['value']}")
        elif expr['type'] == 'name':
            # Variables should already be allocated
            if expr['id'] not in self.qubit_mapping:
                self._allocate_qubit(expr['id'])
        elif expr['type'] == 'binop':
            # Analyze both operands
            self._analyze_expression_requirements(expr['left'])
            self._analyze_expression_requirements(expr['right'])
            # Allocate result qubit
            self._allocate_qubit(f"temp_binop_{len(self.qubit_mapping)}")