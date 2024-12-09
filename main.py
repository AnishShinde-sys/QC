import sys
from src.quantum_translator.translator import QuantumTranslator

def main():
    # Check if file is provided
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file.py>")
        return
    
    # Read input file
    input_file = sys.argv[1]
    try:
        with open(input_file, 'r') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    # Create translator and translate
    translator = QuantumTranslator()
    try:
        circuit = translator.translate(code)
        print("\nGenerated Quantum Circuit:")
        print(circuit)
        print("\nCircuit Diagram:")
        print(circuit.draw())
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 