from .parser import PythonParser
from .circuit_builder import QuantumCircuitBuilder

class QuantumTranslator:
    """Main interface for translating classical code to quantum circuits"""
    
    def __init__(self):
        self.parser = PythonParser()
        self.circuit_builder = QuantumCircuitBuilder()
    
    def translate(self, code: str):
        """
        Translate classical code to a quantum circuit
        
        Args:
            code: Source code string
            
        Returns:
            QuantumCircuit implementation
        """
        # Parse code to IR
        ir = self.parser.parse(code)
        
        # Convert IR to quantum circuit
        circuit = self.circuit_builder.build_circuit(ir)
        
        return circuit 