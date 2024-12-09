from typing import Dict, Any
import ast

class CodeParser:
    """Base parser class for converting source code to AST"""
    
    def parse(self, code: str) -> Dict[str, Any]:
        """
        Parse source code into an intermediate representation
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary containing the IR representation
        """
        raise NotImplementedError

class PythonParser(CodeParser):
    def parse(self, code: str) -> Dict[str, Any]:
        """Parse Python code into IR"""
        tree = ast.parse(code)
        return self._convert_ast_to_ir(tree)
        
    def _convert_ast_to_ir(self, node: ast.AST) -> Dict[str, Any]:
        """Convert AST node to intermediate representation"""
        if isinstance(node, ast.Module):
            return {
                'type': 'module',
                'body': [self._convert_ast_to_ir(stmt) for stmt in node.body]
            }
        elif isinstance(node, ast.For):
            return {
                'type': 'for',
                'target': self._convert_ast_to_ir(node.target),
                'iter': self._convert_ast_to_ir(node.iter),
                'body': [self._convert_ast_to_ir(stmt) for stmt in node.body]
            }
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'range':
                return {
                    'type': 'range',
                    'args': [self._convert_ast_to_ir(arg) for arg in node.args]
                }
            return {
                'type': 'call',
                'function': node.func.id,
                'args': [self._convert_ast_to_ir(arg) for arg in node.args]
            }
        elif isinstance(node, ast.UnaryOp):
            return {
                'type': 'constant',
                'value': -node.operand.n if isinstance(node.op, ast.USub) else node.operand.n
            }
        elif isinstance(node, ast.Assign):
            return {
                'type': 'assign',
                'targets': [self._convert_ast_to_ir(target) for target in node.targets],
                'value': self._convert_ast_to_ir(node.value)
            }
        elif isinstance(node, ast.Name):
            return {
                'type': 'name',
                'id': node.id
            }
        elif isinstance(node, ast.Constant):
            return {
                'type': 'constant',
                'value': node.value
            }
        elif isinstance(node, ast.BinOp):
            return {
                'type': 'binop',
                'left': self._convert_ast_to_ir(node.left),
                'right': self._convert_ast_to_ir(node.right),
                'op': self._convert_operator(node.op),
                'result_type': self._determine_result_type(node)
            }
        elif isinstance(node, ast.Compare):
            return {
                'type': 'compare',
                'left': self._convert_ast_to_ir(node.left),
                'ops': [self._convert_operator(op) for op in node.ops],
                'comparators': [self._convert_ast_to_ir(comp) for comp in node.comparators]
            }
        elif isinstance(node, ast.If):
            return {
                'type': 'if',
                'test': self._convert_ast_to_ir(node.test),
                'body': [self._convert_ast_to_ir(stmt) for stmt in node.body],
                'orelse': [self._convert_ast_to_ir(stmt) for stmt in node.orelse]
            }
        elif isinstance(node, ast.FunctionDef):
            return {
                'type': 'function',
                'name': node.name,
                'args': self._convert_arguments(node.args),
                'body': [self._convert_ast_to_ir(stmt) for stmt in node.body],
                'returns': self._convert_ast_to_ir(node.returns) if node.returns else None
            }
        elif isinstance(node, ast.Return):
            return {
                'type': 'return',
                'value': self._convert_ast_to_ir(node.value) if node.value else None
            }
        elif isinstance(node, ast.arg):
            return {
                'type': 'arg',
                'id': node.arg,
                'annotation': self._convert_ast_to_ir(node.annotation) if node.annotation else None
            }
        elif isinstance(node, ast.Break):
            return {
                'type': 'break'
            }
        # Add more node types as needed
        raise ValueError(f"Unsupported AST node type: {type(node)}") 
    
    def _convert_operator(self, op: ast.operator) -> str:
        """Convert AST operator to IR operator string"""
        op_map = {
            ast.Add: 'add',
            ast.Sub: 'sub',
            ast.Mult: 'mult',
            ast.Div: 'div',
            ast.Mod: 'mod',
            ast.Pow: 'pow',
            ast.BitAnd: 'and',
            ast.BitOr: 'or',
            ast.BitXor: 'xor',
            ast.Eq: 'eq',
            ast.NotEq: 'neq',
            ast.Lt: 'lt',
            ast.LtE: 'lte',
            ast.Gt: 'gt',
            ast.GtE: 'gte',
        }
        op_type = type(op)
        if op_type not in op_map:
            raise ValueError(f"Unsupported operator: {op_type}")
        return op_map[op_type]
    
    def _determine_result_type(self, node: ast.AST) -> str:
        """Determine the type of result for an operation"""
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return 'arithmetic'
            elif isinstance(node.op, ast.Mult):
                return 'multiplication'
            elif isinstance(node.op, ast.BitOr):
                return 'boolean'
        return 'unknown'
    
    def _convert_arguments(self, args: ast.arguments) -> Dict[str, Any]:
        """Convert function arguments to IR"""
        return {
            'args': [self._convert_ast_to_ir(arg) for arg in args.args],
            'defaults': [self._convert_ast_to_ir(default) for default in args.defaults],
            'kwonlyargs': [self._convert_ast_to_ir(arg) for arg in args.kwonlyargs],
            'kw_defaults': [self._convert_ast_to_ir(default) if default else None 
                          for default in args.kw_defaults]
        }