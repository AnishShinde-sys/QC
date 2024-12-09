from src.quantum_translator.translator import QuantumTranslator
from qiskit import QuantumCircuit
from typing import Dict

def test_basic_assignment():
    """Test basic variable assignments"""
    code = """
a = 1
b = 0
c = a
"""
    return code

def test_multiple_assignments():
    """Test multiple assignments and copies"""
    code = """
x = 1
y = x
z = y
w = 0
"""
    return code

def test_binary_values():
    """Test different binary value assignments"""
    code = """
true_bit = 1
false_bit = 0
copy_true = true_bit
copy_false = false_bit
"""
    return code

def test_arithmetic():
    """Test arithmetic operations"""
    code = """
a = 1
b = 1
c = a + b  # Should use quantum addition
d = c - a  # Should use quantum subtraction
"""
    return code

def test_conditional():
    """Test conditional operations with proper control flow"""
    code = """
x = 1
y = 0
if x:
    z = 1
else:
    z = 0
"""
    return code

def test_nested_conditional():
    """Test nested conditional operations"""
    code = """
a = 1
b = 1
if a:
    if b:
        c = 1
    else:
        c = 0
else:
    c = 0
"""
    return code

def test_complex_conditional():
    """Test complex conditional with proper control flow"""
    code = """
x = 1
y = 1
if x:
    if y:
        z = 1
    else:
        z = 0
else:
    z = 0
"""
    return code

def test_loop():
    """Test basic loop operations"""
    code = """
x = 0
for i in range(2):
    x = 1
"""
    return code

def test_loop_with_conditional():
    """Test loop with conditional inside"""
    code = """
result = 0
for i in range(2):
    if i:
        result = 1
"""
    return code

def test_conditional_with_loop():
    """Test conditional with loop inside"""
    code = """
x = 1
if x:
    for i in range(2):
        x = 0
"""
    return code

def test_arithmetic_operations():
    """Test various arithmetic operations"""
    code = """
a = 1
b = 1
sum = a + b
diff = a - b
prod = a * b
quot = a / b
mod = a % b
"""
    return code

def test_complex_arithmetic():
    """Test complex arithmetic expressions"""
    code = """
a = 1
b = 1
c = 1
d = (a + b) * c
e = (a * b) + (c * d)
"""
    return code

def test_bitwise_operations():
    """Test bitwise operations"""
    code = """
a = 1
b = 1
and_result = a & b
or_result = a | b
xor_result = a ^ b
"""
    return code

def test_error_handling():
    """Test error cases"""
    tests = [
        ("Undefined Variable", """
x = y  # y is not defined
"""),
        ("Invalid Operation", """
a = 1
b = a @ 2  # Invalid operator
"""),
        ("Division by Zero", """
a = 1
b = 0
c = a / b
""")
    ]
    return tests

def test_nested_arithmetic():
    """Test deeply nested arithmetic operations"""
    code = """
a = 1
b = 1
c = 1
d = ((a + b) * c) + ((b * c) - a)
"""
    return code

def test_multiple_conditionals():
    """Test multiple conditional statements"""
    code = """
a = 1
b = 0
c = 1
if a:
    x = 1
if b:
    y = 1
if c:
    z = 1
"""
    return code

def test_loop_arithmetic():
    """Test arithmetic operations inside loops"""
    code = """
sum = 0
for i in range(3):
    sum = sum + i
"""
    return code

def test_conditional_arithmetic():
    """Test arithmetic in conditional branches"""
    code = """
a = 1
b = 1
if a:
    c = a + b
else:
    c = a - b
"""
    return code

def test_performance():
    """Performance test cases"""
    import time
    
    def measure_execution(test_name: str, code: str) -> float:
        start_time = time.time()
        translator = QuantumTranslator()
        circuit = translator.translate(code)
        end_time = time.time()
        return end_time - start_time
    
    performance_tests = [
        ("Simple Circuit", """
a = 1
b = 1
c = a + b
"""),
        ("Complex Circuit", """
a = 1
b = 1
for i in range(3):
    if a:
        b = b + 1
    else:
        b = b - 1
"""),
        ("Deep Nesting", """
a = 1
b = 1
c = 1
d = ((a + b) * c) + ((a - b) * (c + 1))
""")
    ]
    
    results = []
    for test_name, code in performance_tests:
        execution_time = measure_execution(test_name, code)
        results.append((test_name, execution_time))
        print(f"\nPerformance Test: {test_name}")
        print(f"Execution Time: {execution_time:.4f} seconds")
    
    return results

def test_circuit_validation():
    """Test circuit properties and constraints"""
    
    def validate_circuit(circuit: QuantumCircuit) -> Dict[str, bool]:
        """Validate various circuit properties"""
        validations = {
            'qubit_count': circuit.num_qubits <= 100,  # Max qubits
            'depth': circuit.depth() <= 1000,  # Max circuit depth
            'has_measurements': any(inst[0].name == 'measure' for inst in circuit.data),
            'valid_gates': all(inst[0].name in ['x', 'cx', 'ccx', 'h', 'measure', 'reset'] 
                             for inst in circuit.data),
            'has_registers': len(circuit.qregs) > 0 and len(circuit.cregs) > 0,
            'proper_initialization': all(len(inst[1]) > 0 for inst in circuit.data)
        }
        return validations
    
    validation_tests = [
        ("Basic Circuit", """
a = 1
b = 0
c = a
"""),
        ("Simple Conditional", """
x = 1
if x:
    y = 1
else:
    y = 0
"""),
        ("Simple Loop", """
x = 0
for i in range(2):
    x = x + 1
""")
    ]
    
    results = []
    translator = QuantumTranslator()
    
    for test_name, code in validation_tests:
        try:
            circuit = translator.translate(code)
            validations = validate_circuit(circuit)
            results.append((test_name, validations))
            
            print(f"\nCircuit Validation: {test_name}")
            for check, passed in validations.items():
                print(f"- {check}: {'✓' if passed else '✗'}")
            
        except Exception as e:
            print(f"\nValidation Failed for {test_name}: {str(e)}")
    
    return results

def validate_test_result(circuit: QuantumCircuit, expected_qubits: int, 
                        expected_depth: int) -> Dict[str, bool]:
    """Validate test result against expected values"""
    validations = {
        'correct_qubit_count': circuit.num_qubits == expected_qubits,
        'depth_within_limit': circuit.depth() <= expected_depth,
        'has_measurements': any(inst[0].name == 'measure' for inst in circuit.data),
        'valid_initialization': all(inst[0].name != 'measure' or len(inst[1]) > 0 
                                  for inst in circuit.data)
    }
    return validations

def run_test(translator, test_name, test_code):
    """Run a specific test with enhanced output"""
    print(f"\n{'='*50}")
    print(f"Running Test: {test_name}")
    print(f"{'='*50}")
    
    # Input code section
    print("\n📝 Input Code:")
    print("```python")
    print(test_code.strip())
    print("```")
    
    try:
        circuit = translator.translate(test_code)
        
        # Circuit information
        print("\n🔧 Circuit Information:")
        print(f"- Qubits: {circuit.num_qubits}")
        print(f"- Depth: {circuit.depth()}")
        print(f"- Gates: {len(circuit.data)}")
        
        # Gate distribution
        print("\n📊 Gate Distribution:")
        gate_counts = {}
        for inst in circuit.data:
            gate_counts[inst[0].name] = gate_counts.get(inst[0].name, 0) + 1
        for gate, count in gate_counts.items():
            print(f"- {gate}: {count}")
        
        # Qubit mapping
        print("\n🔄 Qubit Mapping:")
        for var, qubit in translator.circuit_builder.allocator.variable_to_qubit.items():
            print(f"- {var} → q[{qubit}]")
        
        # Circuit validation
        validations = validate_test_result(circuit, 
                                         expected_qubits=len(translator.circuit_builder.allocator.variable_to_qubit),
                                         expected_depth=50)
        
        print("\n✅ Validation Results:")
        for check, passed in validations.items():
            status = "✓" if passed else "✗"
            print(f"- {check}: {status}")
        
        # Circuit diagram
        print("\n📈 Circuit Diagram:")
        print(circuit.draw())
        
        print("\n🎉 Test Result: Success")
        return True
        
    except Exception as e:
        print("\n❌ Test Result: Failed")
        print(f"Error: {str(e)}")
        raise e

def test_complex_operations():
    """Test complex operations with multiple variables"""
    code = """
a = 1
b = 1
c = (a + b) * (a - b)  # Complex arithmetic
d = c & (a | b)        # Mixed arithmetic and bitwise
e = d + (a * b) - c    # Nested operations
"""
    return code

def test_conditional_chain():
    """Test chained conditional statements"""
    code = """
x = 1
y = 0
z = 1

if x:
    if y:
        result = 1
    elif z:
        result = 2
    else:
        result = 3
else:
    result = 4
"""
    return code

def test_loop_patterns():
    """Test different loop patterns"""
    code = """
sum = 0
product = 1

for i in range(3):
    if i > 0:
        sum = sum + i
        product = product * i
"""
    return code

def test_mixed_operations():
    """Test mixing different types of operations"""
    code = """
a = 1
b = 1
for i in range(2):
    if a & b:
        c = a + b
    else:
        c = a - b
    a = c
"""
    return code

def test_nested_loops():
    """Test nested loop structures"""
    code = """
result = 0
for i in range(2):
    for j in range(2):
        result = result + i + j
"""
    return code

def test_loop_with_break():
    """Test loop with conditional break"""
    code = """
x = 0
for i in range(3):
    if i == 2:
        break
    x = x + 1
"""
    return code

def test_complex_expressions():
    """Test complex expressions with multiple operations"""
    code = """
a = 1
b = 1
c = 1
d = ((a + b) * c) & (a | b)  # Mixed arithmetic and bitwise
e = (a + b) * (c - (a & b))  # Nested with bitwise
f = (d | e) + (a * b)        # Combined operations
"""
    return code

def test_conditional_expressions():
    """Test conditional expressions with complex conditions"""
    code = """
a = 1
b = 1
c = 1

if (a & b) | (b & c):
    result = a + b
else:
    result = b + c

if (a + b) > c:
    x = 1
else:
    x = 0
"""
    return code

def test_variable_reuse():
    """Test reusing variables in different contexts"""
    code = """
x = 1
for i in range(2):
    x = x + i
    if x:
        y = x
    else:
        y = i
    x = y
"""
    return code

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    code = """
# Empty loop
for i in range(0):
    x = 1

# Single iteration
for i in range(1):
    y = 1

# Zero initialization
z = 0
if z:
    w = 1
"""
    return code

def test_quantum_operations():
    """Test quantum-specific operations"""
    code = """
# Superposition
a = 0
b = 1
c = a + b  # Should create superposition

# Entanglement
x = 1
y = x      # Should create entanglement

# Multiple controls
if x & y:
    z = 1
"""
    return code

def test_error_conditions():
    """Test various error conditions"""
    tests = [
        ("Uninitialized Variable", """
x = y + 1  # y is not defined
"""),
        ("Invalid Operation", """
a = 1
b = a ** 2  # Power operation not supported
"""),
        ("Type Mismatch", """
x = 1
y = x + "2"  # String addition not supported
"""),
        ("Invalid Loop Range", """
for i in range(-1):
    x = 1
""")
    ]
    return tests

def print_test_summary(results):
    """Print comprehensive test summary"""
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results['failures']:
        print("\nFailed Tests:")
        for test_name, error in results['failures']:
            print(f"- {test_name}: {error}")
    
    print("\nCircuit Statistics:")
    print(f"- Total Qubits Used: {results['statistics']['total_qubits']}")
    print(f"- Total Gates: {results['statistics']['total_gates']}")
    print(f"- Maximum Circuit Depth: {results['statistics']['max_depth']}")
    print(f"- Average Circuit Depth: {results['statistics']['avg_depth']:.2f}")

def main():
    """Run all tests with enhanced reporting"""
    translator = QuantumTranslator()
    
    # Track test results with more detail
    results = {
        'passed': 0,
        'failed': 0,
        'failures': [],
        'statistics': {
            'total_qubits': 0,
            'total_gates': 0,
            'max_depth': 0,
            'avg_depth': 0.0
        }
    }
    
    # Define all tests
    tests = [
        # Basic tests
        ("Basic Assignment", test_basic_assignment()),
        ("Multiple Assignments", test_multiple_assignments()),
        ("Binary Values", test_binary_values()),
        ("Arithmetic", test_arithmetic()),
        ("Conditional", test_conditional()),
        
        # Complex tests
        ("Nested Conditional", test_nested_conditional()),
        ("Complex Conditional", test_complex_conditional()),
        ("Loop", test_loop()),
        ("Nested Loops", test_nested_loops()),
        ("Loop with Break", test_loop_with_break()),
        
        # Advanced tests
        ("Complex Expressions", test_complex_expressions()),
        ("Conditional Expressions", test_conditional_expressions()),
        ("Variable Reuse", test_variable_reuse()),
        ("Edge Cases", test_edge_cases()),
        ("Quantum Operations", test_quantum_operations())
    ]
    
    # Add error test cases
    for error_test_name, error_test_code in test_error_conditions():
        tests.append((f"Error - {error_test_name}", error_test_code))
    
    # Run all tests
    for test_name, test_code in tests:
        try:
            run_test(translator, test_name, test_code)
            results['passed'] += 1
            
            # Update statistics if test passed
            if isinstance(test_code, str):  # Skip error test cases
                circuit = translator.translate(test_code)
                results['statistics']['total_qubits'] += circuit.num_qubits
                results['statistics']['total_gates'] += len(circuit.data)
                results['statistics']['max_depth'] = max(
                    results['statistics']['max_depth'], 
                    circuit.depth()
                )
                
        except Exception as e:
            results['failed'] += 1
            results['failures'].append((test_name, str(e)))
    
    # Calculate averages
    total_tests = len([t for t in tests if isinstance(t[1], str)])
    if total_tests > 0:
        results['statistics']['avg_depth'] = results['statistics']['total_gates'] / total_tests
    
    # Print comprehensive summary
    print_test_summary(results)

if __name__ == "__main__":
    main() 