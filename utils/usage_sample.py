# =============================================================================
# SIMPLE TEST OF ENHANCED EXECUTOR
# =============================================================================

# First, make sure you have the executor class available
# You can copy the DataAnalysisCodeExecutor and EnhancedDataAnalysisExecutor 
# classes from the previous artifact

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

# Copy the executor classes here or import them
class DataAnalysisCodeExecutor:
    """Basic executor - copy from previous artifact"""
    def __init__(self, timeout=60, memory_limit_mb=500):
        self.timeout = timeout
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'sklearn', 'scipy', 'statsmodels', 'datetime', 'time',
            'json', 're', 'math', 'statistics', 'itertools', 
            'functools', 'collections', 'pathlib', 'os', 'sys', 'warnings'
        }
    
    def execute(self, code, context=None):
        execution_globals = self._create_safe_globals()
        execution_locals = context.copy() if context else {}
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'locals': {}
        }
        
        start_time = time.time()
        
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            exec(code, execution_globals, execution_locals)
            
            result['success'] = True
            result['locals'] = self._filter_locals(execution_locals)
            result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {str(e)}'
            result['execution_time'] = time.time() - start_time
            
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result['output'] = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            if stderr_output and not result['error']:
                result['error'] = stderr_output
        
        return result
    
    def _create_safe_globals(self):
        safe_builtins = {
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
            'any': any, 'all': all,
            'Exception': Exception, 'ValueError': ValueError,
            'TypeError': TypeError, 'KeyError': KeyError,
            'None': None, 'True': True, 'False': False,
        }
        
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split('.')[0]
            if base_module not in self.allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)
        
        return {
            '__builtins__': safe_builtins,
            '__import__': safe_import,
            '__name__': '__main__',
        }
    
    def _filter_locals(self, locals_dict):
        filtered = {}
        for key, value in locals_dict.items():
            if key.startswith('_'):
                continue
            if isinstance(value, types.ModuleType):
                continue
            if isinstance(value, types.FunctionType):
                continue
            try:
                str(value)
                filtered[key] = value
            except:
                filtered[key] = f"<{type(value).__name__} object>"
        return filtered

class EnhancedDataAnalysisExecutor(DataAnalysisCodeExecutor):
    """Enhanced version with context preservation"""
    
    def __init__(self, timeout=120, memory_limit_mb=1000):
        super().__init__(timeout, memory_limit_mb)
        self.execution_history = []
        self.data_cache = {}
    
    def execute_with_context(self, code, preserve_context=True):
        context = self.data_cache if preserve_context else None
        result = self.execute(code, context)
        
        if result['success'] and preserve_context:
            self.data_cache.update(result['locals'])
        
        self.execution_history.append({
            'code': code,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_variable(self, var_name):
        return self.data_cache.get(var_name)
    
    def list_variables(self):
        return list(self.data_cache.keys())
    
    def clear_context(self):
        self.data_cache.clear()
        gc.collect()
    
    def get_execution_summary(self):
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': sum(1 for h in self.execution_history if h['result']['success']),
            'total_execution_time': sum(h['result']['execution_time'] for h in self.execution_history),
            'current_variables': len(self.data_cache)
        }

# =============================================================================
# SIMPLE USAGE TEST
# =============================================================================

def test_enhanced_executor():
    """Simple test showing the key enhanced features"""
    
    print("=== Enhanced Executor Test ===")
    
    # Create executor
    executor = EnhancedDataAnalysisExecutor(timeout=30)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    code1 = """
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'product': ['A', 'B', 'C'] * 10,
    'sales': np.random.randint(50, 500, 30),
    'price': np.random.uniform(10, 100, 30)
})

print(f"Data shape: {data.shape}")
print("First 5 rows:")
print(data.head())
"""
    
    result1 = executor.execute_with_context(code1, preserve_context=True)
    
    if result1['success']:
        print("✅ Data loaded successfully!")
        print("Output:", result1['output'])
        print("Variables created:", list(result1['locals'].keys()))
    else:
        print("❌ Failed to load data:")
        print("Error:", result1['error'])
        return
    
    # Step 2: Analyze data (using the 'data' variable from step 1)
    print("\n2. Analyzing data...")
    code2 = """
# The 'data' variable is available from the previous step!
sales_by_product = data.groupby('product')['sales'].sum()
print("Sales by product:")
print(sales_by_product)

best_product = sales_by_product.idxmax()
best_sales = sales_by_product.max()
print(f"\\nBest product: {best_product} with {best_sales} total sales")
"""
    
    result2 = executor.execute_with_context(code2, preserve_context=True)
    
    if result2['success']:
        print("✅ Analysis completed!")
        print("Output:", result2['output'])
    else:
        print("❌ Analysis failed:")
        print("Error:", result2['error'])
    
    # Step 3: Show context management
    print("\n3. Context Management...")
    print("Current variables:", executor.list_variables())
    
    # Get a specific variable
    data_var = executor.get_variable('data')
    if data_var is not None:
        print(f"Retrieved 'data' variable: shape {data_var.shape}")
    
    # Show execution summary
    summary = executor.get_execution_summary()
    print("\nExecution Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Step 4: Clear context and start fresh
    print("\n4. Clearing context...")
    executor.clear_context()
    print("Variables after clearing:", executor.list_variables())
    
    return executor

def test_error_handling():
    """Test error handling"""
    
    print("\n=== Error Handling Test ===")
    
    executor = EnhancedDataAnalysisExecutor()
    
    # Try to use a variable that doesn't exist
    bad_code = """
print(nonexistent_variable)
"""
    
    result = executor.execute_with_context(bad_code)
    
    if not result['success']:
        print("✅ Error caught successfully!")
        print("Error:", result['error'])
    else:
        print("❌ Expected an error but code succeeded")

if __name__ == "__main__":
    print("TESTING ENHANCED DATA ANALYSIS EXECUTOR")
    print("=" * 50)
    
    # Test the enhanced features
    executor = test_enhanced_executor()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("Test completed! The enhanced executor provides:")
    print("✓ Context preservation between executions")
    print("✓ Variable management and inspection")
    print("✓ Execution history tracking")
    print("✓ Error handling and recovery")