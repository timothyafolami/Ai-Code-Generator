# =============================================================================
# CUSTOM CODE EXECUTOR FOR AI DATA ANALYSIS GENERATOR
# =============================================================================

import sys
import ast
import time
import traceback
import types
import importlib
from io import StringIO
import contextlib
import gc
import threading
import signal

# =============================================================================
# 1. MAIN CODE EXECUTOR CLASS
# =============================================================================

class DataAnalysisCodeExecutor:
    """
    Advanced code executor designed specifically for AI-generated data analysis code.
    Provides safety, output capture, timeout handling, and execution monitoring.
    """
    
    def __init__(self, timeout=60, memory_limit_mb=500):
        self.timeout = timeout
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.allowed_modules = {
            # Data manipulation
            'pandas', 'numpy', 'polars', 'dask',
            # Visualization
            'matplotlib', 'seaborn', 'plotly', 'altair', 'bokeh',
            # Machine Learning
            'sklearn', 
            # Statistical Analysis
            'scipy', 'statsmodels', 'pingouin',
            # Time Series
            'prophet', 'sktime', 'tsfresh',
            # Utilities
            'datetime', 'time', 'json', 're', 'math', 'statistics',
            'itertools', 'functools', 'collections', 'pathlib',
            'os', 'sys', 'warnings', 'pickle', 'typing'
        }
    
    def execute(self, code, context=None):
        """
        Execute code with full monitoring and safety features
        
        Args:
            code (str): Python code to execute
            context (dict): Optional pre-existing variables/context
            
        Returns:
            dict: Execution results with success status, output, errors, etc.
        """
        
        # Validate code first
        validation = self.validate_code(code)
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error'],
                'output': '',
                'execution_time': 0,
                'locals': {},
                'validation': validation
            }
        
        # Setup execution environment
        execution_globals = self._create_safe_globals()
        execution_locals = context.copy() if context else {}
        
        # Setup output capture
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'locals': {},
            'validation': validation,
            'memory_used': 0,
            'plots_created': []
        }
        
        start_time = time.time()
        
        try:
            # Setup timeout handler
            if self.timeout:
                self._setup_timeout()
            
            # Redirect output streams
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            # Execute the code
            exec(code, execution_globals, execution_locals)
            
            # Successful execution
            result['success'] = True
            result['locals'] = self._filter_locals(execution_locals)
            result['execution_time'] = time.time() - start_time
            
        except TimeoutError:
            result['error'] = f'Code execution timed out after {self.timeout} seconds'
            result['execution_time'] = self.timeout
            
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {str(e)}'
            result['traceback'] = traceback.format_exc()
            result['execution_time'] = time.time() - start_time
            
        finally:
            # Restore output streams
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
            # Cancel timeout if set
            if self.timeout:
                self._cancel_timeout()
            
            # Capture outputs
            result['output'] = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            if stderr_output and not result['error']:
                result['error'] = stderr_output
            
            # Memory cleanup
            gc.collect()
        
        return result
    
    def validate_code(self, code):
        """
        Validate code syntax and check for security issues
        
        Args:
            code (str): Python code to validate
            
        Returns:
            dict: Validation results
        """
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            analysis = {
                'valid': True,
                'imports': [],
                'functions_defined': [],
                'variables_assigned': [],
                'function_calls': [],
                'warnings': [],
                'forbidden_operations': []
            }
            
            # Analyze the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        analysis['imports'].append(alias.name)
                        if module_name not in self.allowed_modules:
                            analysis['warnings'].append(f"Module '{alias.name}' not in allowed list")
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    module_name = module.split('.')[0]
                    analysis['imports'].append(module)
                    if module_name not in self.allowed_modules and module_name:
                        analysis['warnings'].append(f"Module '{module}' not in allowed list")
                
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions_defined'].append(node.name)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables_assigned'].append(target.id)
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        analysis['function_calls'].append(func_name)
                        
                        # Check for forbidden operations
                        forbidden = ['exec', 'eval', 'compile', '__import__', 'open']
                        if func_name in forbidden:
                            analysis['forbidden_operations'].append(func_name)
            
            # Check for forbidden operations
            if analysis['forbidden_operations']:
                analysis['valid'] = False
                analysis['error'] = f"Forbidden operations detected: {', '.join(analysis['forbidden_operations'])}"
            
            return analysis
            
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f'Syntax Error: {str(e)}',
                'line': getattr(e, 'lineno', None),
                'offset': getattr(e, 'offset', None)
            }
    
    def _create_safe_globals(self):
        """Create a safe global execution environment"""
        
        # Start with minimal builtins
        safe_builtins = {
            # Basic types
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            
            # Basic functions
            'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'sum': sum, 'min': max, 'max': max, 'abs': abs, 'round': round,
            'any': any, 'all': all,
            
            # Exceptions
            'Exception': Exception, 'ValueError': ValueError,
            'TypeError': TypeError, 'KeyError': KeyError,
            'IndexError': IndexError, 'AttributeError': AttributeError,
            
            # Special
            'None': None, 'True': True, 'False': False,
        }
        
        # Create safe import function
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split('.')[0]
            if base_module not in self.allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed in this environment")
            return __import__(name, globals, locals, fromlist, level)
        
        return {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
        }
    
    def _filter_locals(self, locals_dict):
        """Filter local variables to return only relevant results"""
        
        filtered = {}
        for key, value in locals_dict.items():
            # Skip private variables and functions
            if key.startswith('_'):
                continue
            
            # Skip modules
            if isinstance(value, types.ModuleType):
                continue
            
            # Skip functions (unless specifically requested)
            if isinstance(value, types.FunctionType):
                continue
            
            # Include everything else
            try:
                # Test if the value is serializable (roughly)
                str(value)
                filtered[key] = value
            except:
                filtered[key] = f"<{type(value).__name__} object>"
        
        return filtered
    
    def _setup_timeout(self):
        """Setup timeout handler using threading"""
        
        def timeout_handler():
            raise TimeoutError(f"Code execution exceeded {self.timeout} seconds")
        
        self.timeout_timer = threading.Timer(self.timeout, timeout_handler)
        self.timeout_timer.start()
    
    def _cancel_timeout(self):
        """Cancel the timeout timer"""
        
        if hasattr(self, 'timeout_timer'):
            self.timeout_timer.cancel()

# =============================================================================
# 2. ENHANCED EXECUTOR WITH DATA ANALYSIS FEATURES
# =============================================================================

class EnhancedDataAnalysisExecutor(DataAnalysisCodeExecutor):
    """
    Enhanced version with specific features for data analysis workflows
    """
    
    def __init__(self, timeout=120, memory_limit_mb=1000):
        super().__init__(timeout, memory_limit_mb)
        self.execution_history = []
        self.data_cache = {}
    
    def execute_with_context(self, code, preserve_context=True):
        """
        Execute code while preserving context between executions
        
        Args:
            code (str): Python code to execute
            preserve_context (bool): Whether to maintain variables between executions
            
        Returns:
            dict: Execution results
        """
        
        # Use cached context if preserving
        context = self.data_cache if preserve_context else None
        
        result = self.execute(code, context)
        
        # Update cache with new variables
        if result['success'] and preserve_context:
            self.data_cache.update(result['locals'])
        
        # Store in history
        self.execution_history.append({
            'code': code,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_variable(self, var_name):
        """Get a specific variable from the execution context"""
        return self.data_cache.get(var_name)
    
    def list_variables(self):
        """List all variables in the current context"""
        return list(self.data_cache.keys())
    
    def clear_context(self):
        """Clear the execution context"""
        self.data_cache.clear()
        gc.collect()
    
    def get_execution_summary(self):
        """Get summary of all executions"""
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': sum(1 for h in self.execution_history if h['result']['success']),
            'total_execution_time': sum(h['result']['execution_time'] for h in self.execution_history),
            'current_variables': len(self.data_cache)
        }

# =============================================================================
# 3. USAGE EXAMPLES AND TESTING
# =============================================================================

def test_code_executor():
    """Test the code executor with various scenarios"""
    
    executor = EnhancedDataAnalysisExecutor(timeout=30)
    
    # Test 1: Basic data analysis
    print("=== Test 1: Basic Data Analysis ===")
    code1 = """
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100)
})

print("Dataset created:")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\\nFirst few rows:")
print(data.head())

# Basic statistics
stats = data.describe()
print("\\nBasic statistics:")
print(stats)
"""
    
    result1 = executor.execute_with_context(code1)
    print(f"Success: {result1['success']}")
    print(f"Execution time: {result1['execution_time']:.2f}s")
    print(f"Variables created: {list(result1['locals'].keys())}")
    print("Output (first 500 chars):")
    print(result1['output'][:500])
    
    # Test 2: Using previous context
    print("\n=== Test 2: Using Previous Context ===")
    code2 = """
# Use the data from previous execution
correlation = data[['A', 'B']].corr()
print("Correlation matrix:")
print(correlation)

# Group by categorical column
grouped = data.groupby('C').mean()
print("\\nMean by category:")
print(grouped)
"""
    
    result2 = executor.execute_with_context(code2)
    print(f"Success: {result2['success']}")
    print("Output:")
    print(result2['output'])
    
    # Test 3: Error handling
    print("\n=== Test 3: Error Handling ===")
    code3 = """
# This will cause an error
invalid_operation = data['NonExistentColumn'].mean()
"""
    
    result3 = executor.execute_with_context(code3)
    print(f"Success: {result3['success']}")
    print(f"Error: {result3['error']}")
    
    # Test 4: Code validation
    print("\n=== Test 4: Code Validation ===")
    invalid_code = """
import subprocess
subprocess.run(['rm', '-rf', '/'])  # Dangerous!
"""
    
    validation = executor.validate_code(invalid_code)
    print(f"Valid: {validation['valid']}")
    print(f"Warnings: {validation.get('warnings', [])}")
    
    # Summary
    print("\n=== Execution Summary ===")
    summary = executor.get_execution_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("Custom Data Analysis Code Executor")
    print("=" * 50)
    test_code_executor()